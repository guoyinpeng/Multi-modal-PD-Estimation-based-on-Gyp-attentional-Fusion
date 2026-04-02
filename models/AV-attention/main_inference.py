import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from datetime import datetime
from autolab_core import YamlConfig

from utils import *
from models.fusion import PDAFFusion, SimpleConcatFusion
from models.bypass_bn import enable_running_stats, disable_running_stats


def _fusion_forward(visual_net, audio_net, fusion_net, evaluator, input_batch, args, use_vector_fusion):
    B, T, Fv, C = input_batch['visual'].shape
    visual_input = input_batch['visual'].permute(0, 3, 2, 1).contiguous()
    visual_vec = visual_net(visual_input.to(args.device), return_sequence=False)

    B, Fa, Ta = input_batch['audio'].shape
    audio_input = input_batch['audio'].view(B, Fa, Ta)
    audio_vec = audio_net(audio_input.to(args.device), return_sequence=False)

    if use_vector_fusion:
        fused = fusion_net(visual_vec, audio_vec)
    else:
        fused_tensor = torch.stack([visual_vec, audio_vec], dim=1).unsqueeze(1)
        fused = fusion_net(fused_tensor)
        fused = fused.view(B, -1)

    return evaluator(fused)


def main(dataloaders, visual_net, audio_net, fusion_net, evaluator, base_logger, writer,
         config, args, model_type, ckpt_path, fold_idx=None, run_trace_dir=None):
    def _is_vec_fusion(net):
        m = net.module if isinstance(net, nn.DataParallel) else net
        return isinstance(m, (PDAFFusion, SimpleConcatFusion))

    use_vector_fusion = _is_vec_fusion(fusion_net)

    model_parameters = (
        list(evaluator.parameters())
        + list(visual_net.parameters())
        + list(audio_net.parameters())
        + list(fusion_net.parameters())
    )
    optimizer, scheduler = get_optimizer_scheduler(model_parameters, config['OPTIMIZER'], config['SCHEDULER'])

    assert '_train_labels' in dataloaders, 'dataloaders 需包含 _train_labels（训练折标签，用于 CB-Focal 计数）'
    samples_per_cls = get_samples_per_cls(dataloaders['_train_labels'], config['EVALUATOR'])
    criterion = get_criterion(config['CRITERION'], args, samples_per_cls=samples_per_cls)

    patience = config.get('EARLY_STOP_PATIENCE', 15)
    best_val_loss = float('inf')
    stall = 0

    test_best_f1_score = 0
    test_epoch_best_f1 = 0
    test_best_acc = 0
    test_epoch_best_acc = 0

    best_fold_result = {
        'fold': fold_idx,
        'acc': 0.0,
        'tpr': 0.0,
        'tnr': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'epoch': 0
    }

    for epoch in range(config['EPOCHS']):
        epoch_t0 = time.time()
        trace_row = {}
        if fold_idx is not None:
            log_and_print(base_logger,
                          f'Fold: {fold_idx} | Epoch: {epoch}  当前最佳 F1: {test_best_f1_score} （Epoch {test_epoch_best_f1}）')
            log_and_print(base_logger,
                          f'Fold: {fold_idx} | Epoch: {epoch}  当前最佳 Acc： {test_best_acc} （Epoch {test_epoch_best_acc}）')
        else:
            log_and_print(base_logger,
                          f'Epoch: {epoch}  当前最佳 F1: {test_best_f1_score} （Epoch {test_epoch_best_f1}）')
            log_and_print(base_logger,
                          f'Epoch: {epoch}  当前最佳 Acc： {test_best_acc} （Epoch {test_epoch_best_acc}）')

        for mode in ['train', 'val', 'test']:
            mode_start_time = time.time()

            pd_binary_gt = []
            pd_binary_pred = []

            if mode == 'train':
                visual_net.train()
                audio_net.train()
                fusion_net.train()
                evaluator.train()
                torch.set_grad_enabled(True)
            else:
                visual_net.eval()
                audio_net.eval()
                fusion_net.eval()
                evaluator.eval()
                torch.set_grad_enabled(False)

            total_loss = 0
            log_interval_loss = 0
            log_interval = 10
            batch_number = 0
            n_batches = len(dataloaders[mode])
            batches_start_time = time.time()

            for data in tqdm(dataloaders[mode]):
                batch_size = data['ID'].size(0)

                pd_binary_gt.extend(data['pd_binary_gt'].numpy().astype(float))

                def model_processing(inp):
                    return _fusion_forward(
                        visual_net, audio_net, fusion_net, evaluator, inp, args, use_vector_fusion
                    )

                if mode == 'train':
                    gt = get_gt(data, config['EVALUATOR']['PREDICT_TYPE'])

                    if config['CRITERION'].get('USE_WEIGHTS'):
                        config['CRITERION']['WEIGHTS'] = get_crossentropy_weights(gt, config['EVALUATOR'])

                    if config['OPTIMIZER'].get('USE_SAM'):
                        models = [visual_net, audio_net, fusion_net, evaluator]
                        for model in models:
                            enable_running_stats(model)

                        probs = model_processing(data)
                        gt_t = gt.float().to(args.device)
                        if probs.dim() == 2 and probs.size(1) == 1:
                            probs = probs.view(-1)
                        loss = criterion(probs, gt_t)
                        loss.backward()
                        optimizer.first_step(zero_grad=True)

                        for model in models:
                            disable_running_stats(model)

                        probs2 = model_processing(data)
                        if probs2.dim() == 2 and probs2.size(1) == 1:
                            probs2 = probs2.view(-1)
                        compute_loss(criterion, probs2, gt_t, args).backward()
                        optimizer.second_step(zero_grad=True)
                    else:
                        probs = model_processing(data)
                        gt_t = gt.float().to(args.device)
                        if probs.dim() == 2 and probs.size(1) == 1:
                            probs = probs.view(-1)
                        loss = criterion(probs, gt_t)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    pd_binary = (torch.sigmoid(probs) > 0.5).float().to(args.device)
                    pd_binary_pred.extend([pd_binary[i].item() for i in range(batch_size)])
                    total_loss += loss.item()
                    log_interval_loss += loss.item()
                    if batch_number % log_interval == 0 and batch_number > 0:
                        lr = scheduler.get_last_lr()[0]
                        ms_per_batch = (time.time() - batches_start_time) * 1000 / log_interval
                        current_loss = log_interval_loss / log_interval
                        print(f'| epoch {epoch:3d} | {mode} | {batch_number:3d}/{n_batches:3d} batches | '
                              f'LR {lr:7.6f} | ms/batch {ms_per_batch:5.2f} | loss {current_loss:8.5f} |')

                        writer.add_scalar(
                            'Loss_per_{}_batches/{}'.format(log_interval, mode),
                            current_loss,
                            epoch * n_batches + batch_number
                        )

                        log_interval_loss = 0
                        batches_start_time = time.time()
                else:
                    with torch.no_grad():
                        probs = model_processing(data)
                        gt = get_gt(data, config['EVALUATOR']['PREDICT_TYPE']).float().to(args.device)
                        if probs.dim() == 2 and probs.size(1) == 1:
                            probs = probs.view(-1)
                        loss = criterion(probs, gt)
                        total_loss += loss.item()

                    pd_binary = (torch.sigmoid(probs) > 0.5).float().to(args.device)
                    pd_binary_pred.extend([pd_binary[i].item() for i in range(batch_size)])

                batch_number += 1

            average_loss = total_loss / max(n_batches, 1)
            lr = scheduler.get_last_lr()[0]
            s_per_mode = time.time() - mode_start_time
            accuracy, correct_number = get_accuracy(pd_binary_gt, pd_binary_pred)

            print('-' * 110)
            msg = ('  {0}:\n  | time: {1:8.3f}s | LR: {2:7.6f} | Average Loss: {3:8.5f} | Accuracy: {4:5.2f}%'
                   ' ({5}/{6}) |').format(mode, s_per_mode, lr, average_loss, accuracy * 100, correct_number,
                                          len(pd_binary_gt))
            log_and_print(base_logger, msg)
            print('-' * 110)

            writer.add_scalar('Loss_per_epoch/{}'.format(mode), average_loss, epoch)
            writer.add_scalar('Accuracy/{}'.format(mode), accuracy * 100, epoch)
            writer.add_scalar('Learning_rate/{}'.format(mode), lr, epoch)

            log_and_print(base_logger, '  Output Scores:')

            [[tp, fp], [fn, tn]] = standard_confusion_matrix(pd_binary_gt, pd_binary_pred)
            msg = (f'  - Confusion Matrix:\n'
                   '    -----------------------\n'
                   f'    | TP: {tp:4.0f} | FP: {fp:4.0f} |\n'
                   '    -----------------------\n'
                   f'    | FN: {fn:4.0f} | TN: {tn:4.0f} |\n'
                   '    -----------------------')
            log_and_print(base_logger, msg)

            tpr, tnr, precision, recall, f1_score = get_classification_scores(pd_binary_gt, pd_binary_pred)
            msg = ('  - Classification:\n'
                   '      TPR/Sensitivity: {0:6.4f}\n'
                   '      TNR/Specificity: {1:6.4f}\n'
                   '      Precision: {2:6.4f}\n'
                   '      Recall: {3:6.4f}\n'
                   '      F1-score: {4:6.4f}').format(tpr, tnr, precision, recall, f1_score)
            log_and_print(base_logger, msg)

            trace_row[f'{mode}_loss'] = round(float(average_loss), 6)
            trace_row[f'{mode}_acc'] = round(float(accuracy), 6)
            if mode == 'test':
                trace_row['test_f1'] = round(float(f1_score), 6)
                trace_row['test_tpr'] = round(float(tpr), 6)
                trace_row['test_tnr'] = round(float(tnr), 6)

            if mode == 'val':
                if average_loss < best_val_loss:
                    best_val_loss = average_loss
                    stall = 0
                    save_model_weights(
                        visual_net, audio_net, fusion_net, evaluator,
                        epoch, model_type, ckpt_path,
                        best_metric='val_loss', value=best_val_loss
                    )
                else:
                    stall += 1

            if mode == 'test':
                if accuracy > test_best_acc:
                    test_best_acc = accuracy
                    test_epoch_best_acc = epoch
                    save_model_weights(
                        visual_net, audio_net, fusion_net, evaluator,
                        epoch, model_type, ckpt_path,
                        best_metric='acc', value=test_best_acc
                    )

                if f1_score > test_best_f1_score:
                    test_best_f1_score = f1_score
                    test_epoch_best_f1 = epoch

                    best_fold_result = {
                        'fold': fold_idx,
                        'acc': accuracy,
                        'tpr': tpr,
                        'tnr': tnr,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1_score,
                        'epoch': epoch
                    }

                    save_model_weights(
                        visual_net, audio_net, fusion_net, evaluator,
                        epoch, model_type, ckpt_path,
                        best_metric='f1', value=test_best_f1_score
                    )

        trace_row['ts'] = datetime.now().isoformat()
        trace_row['fold'] = fold_idx if fold_idx is not None else -1
        trace_row['epoch'] = epoch
        trace_row['lr'] = round(float(scheduler.get_last_lr()[0]), 8)
        trace_row['epoch_sec'] = round(time.time() - epoch_t0, 3)
        if run_trace_dir:
            append_epoch_trace_row(os.path.join(run_trace_dir, 'epoch_metrics.csv'), trace_row)

        scheduler.step()

        if stall >= patience:
            log_and_print(base_logger, f'Early stopping at epoch {epoch} (patience={patience}, val loss 无改善).')
            break

    return best_fold_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help="配置文件路径", required=False,
                        default='config/config_inference.yaml')
    parser.add_argument('--device', type=str, help="设备类型: 'cpu' or 'cuda' (GPU)", required=False,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--gpu', type=str, help='使用的 GPU 设备编号', required=False, default='2, 3')
    parser.add_argument('--save', type=bool, help='是否保存最佳模型', required=False, default=False)
    args = parser.parse_args()

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = YamlConfig(args.config_file)

    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(config['CKPTS_DIR'], exist_ok=True)
    os.makedirs(os.path.join(config['CKPTS_DIR'], config['TYPE']), exist_ok=True)
    os.makedirs(config['MODEL']['WEIGHTS']['PATH'], exist_ok=True)

    print('=' * 40)
    config.save(os.path.join(config['OUTPUT_DIR'], config['SAVE_CONFIG_NAME']))
    print('=' * 40)

    init_seed(config['MANUAL_SEED'])

    model_type = config['TYPE']
    all_fold_results = []

    n_splits = 5

    for fold_idx in range(n_splits):
        print('=' * 80)
        print(f'开始运行 Fold {fold_idx + 1}/{n_splits}')
        print('=' * 80)

        fold_output_dir = os.path.join(config['OUTPUT_DIR'], f'fold_{fold_idx + 1}')
        fold_ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'], f'fold_{fold_idx + 1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        os.makedirs(fold_ckpt_path, exist_ok=True)

        file_name = os.path.join(fold_output_dir, '{}_fold{}.log'.format(config['TYPE'], fold_idx + 1))
        base_logger = get_logger(file_name, '{} | Fold {}'.format(config['LOG_TITLE'], fold_idx + 1))
        writer = SummaryWriter(os.path.join(fold_output_dir, 'runs'))

        dataloaders, fold_info = get_dataloaders(
            config['DATA'],
            fold_idx=fold_idx,
            n_splits=n_splits,
            seed=config['MANUAL_SEED']
        )
        dataloaders['_train_labels'] = fold_info['train_labels']

        log_and_print(base_logger, f"Fold {fold_idx + 1} 信息: {fold_info}")
        print(f"Fold {fold_idx + 1} 信息: {fold_info}")

        write_run_manifest_json(
            os.path.join(fold_output_dir, 'run_manifest.json'),
            {
                'started_at': datetime.now().isoformat(),
                'config_file': os.path.abspath(args.config_file),
                'model_type': config['TYPE'],
                'fold': fold_idx + 1,
                'n_splits': n_splits,
                'device': str(args.device),
                'cuda_available': bool(torch.cuda.is_available()),
                'torch': torch.__version__,
                'seed': config['MANUAL_SEED'],
                'output_dir': os.path.abspath(fold_output_dir),
                'ckpt_dir': os.path.abspath(fold_ckpt_path),
            },
        )

        visual_net, audio_net, fusion_net, evaluator = get_models(
            config['MODEL'], args, model_type, fold_ckpt_path
        )

        fold_result = main(
            dataloaders,
            visual_net,
            audio_net,
            fusion_net,
            evaluator,
            base_logger,
            writer,
            config['MODEL'],
            args,
            model_type,
            fold_ckpt_path,
            fold_idx=fold_idx + 1,
            run_trace_dir=fold_output_dir,
        )

        all_fold_results.append(fold_result)
        writer.close()

        print(f"Fold {fold_idx + 1} 结果: {fold_result}")

    acc_list = [x['acc'] for x in all_fold_results]
    tpr_list = [x['tpr'] for x in all_fold_results]
    tnr_list = [x['tnr'] for x in all_fold_results]
    f1_list = [x['f1'] for x in all_fold_results]

    def mean_std(x):
        x = np.array(x, dtype=float)
        mean = np.mean(x)
        std = np.std(x, ddof=1) if len(x) > 1 else 0.0
        return mean, std

    acc_mean, acc_std = mean_std(acc_list)
    tpr_mean, tpr_std = mean_std(tpr_list)
    tnr_mean, tnr_std = mean_std(tnr_list)
    f1_mean, f1_std = mean_std(f1_list)

    print('\n' + '=' * 80)
    print('5-Fold Cross-Validation Results')
    print('=' * 80)
    print(f'Accuracy : {acc_mean * 100:.2f} ± {acc_std * 100:.2f}')
    print(f'TPR      : {tpr_mean * 100:.2f} ± {tpr_std * 100:.2f}')
    print(f'TNR      : {tnr_mean * 100:.2f} ± {tnr_std * 100:.2f}')
    print(f'F1-score : {f1_mean:.4f} ± {f1_std:.4f}')

    results_df = pd.DataFrame(all_fold_results)
    results_csv_path = os.path.join(config['OUTPUT_DIR'], 'five_fold_results.csv')
    results_df.to_csv(results_csv_path, index=False)

    summary_txt_path = os.path.join(config['OUTPUT_DIR'], 'five_fold_summary.txt')
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write('5-Fold Cross-Validation Results\n')
        f.write('=' * 80 + '\n')
        f.write(f'Accuracy : {acc_mean * 100:.2f} ± {acc_std * 100:.2f}\n')
        f.write(f'TPR      : {tpr_mean * 100:.2f} ± {tpr_std * 100:.2f}\n')
        f.write(f'TNR      : {tnr_mean * 100:.2f} ± {tnr_std * 100:.2f}\n')
        f.write(f'F1-score : {f1_mean:.4f} ± {f1_std:.4f}\n\n')
        f.write(results_df.to_string(index=False))

    print(f'\n每折结果已保存到: {results_csv_path}')
    print(f'汇总结果已保存到: {summary_txt_path}')

    write_run_session_summary_json(
        os.path.join(config['OUTPUT_DIR'], 'run_session_summary.json'),
        {
            'finished_at': datetime.now().isoformat(),
            'config_file': os.path.abspath(args.config_file),
            'model_type': config['TYPE'],
            'n_splits': n_splits,
            'mean_acc': acc_mean,
            'std_acc': acc_std,
            'mean_f1': f1_mean,
            'std_f1': f1_std,
            'results_csv': os.path.abspath(results_csv_path),
            'summary_txt': os.path.abspath(summary_txt_path),
            'per_fold_epoch_traces': [
                os.path.abspath(os.path.join(config['OUTPUT_DIR'], f'fold_{i + 1}', 'epoch_metrics.csv'))
                for i in range(n_splits)
            ],
        },
    )
    print(f"会话汇总: {os.path.join(config['OUTPUT_DIR'], 'run_session_summary.json')}")
