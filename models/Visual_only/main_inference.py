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
from models.bypass_bn import enable_running_stats, disable_running_stats


def _forward_visual(visual_net, evaluator, data, args):
    B, T, Fv, C = data['visual'].shape
    visual_input = data['visual'].permute(0, 3, 2, 1).contiguous()
    feat = visual_net(visual_input.to(args.device))
    return evaluator(feat)


def main(dataloaders, visual_net, evaluator, base_logger, writer, config, args, model_type, ckpt_path, fold_idx=None,
         run_trace_dir=None):
    model_parameters = list(visual_net.parameters()) + list(evaluator.parameters())
    optimizer, scheduler = get_optimizer_scheduler(model_parameters, config['OPTIMIZER'], config['SCHEDULER'])

    assert '_train_labels' in dataloaders
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
        'fold': fold_idx, 'acc': 0.0, 'tpr': 0.0, 'tnr': 0.0,
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'epoch': 0
    }

    for epoch in range(config['EPOCHS']):
        epoch_t0 = time.time()
        trace_row = {}
        if fold_idx is not None:
            log_and_print(base_logger,
                          f'Fold: {fold_idx} | Epoch: {epoch}  最佳 F1: {test_best_f1_score} （Epoch {test_epoch_best_f1}）')
        else:
            log_and_print(base_logger, f'Epoch: {epoch}  最佳 F1: {test_best_f1_score}')

        for mode in ['train', 'val', 'test']:
            mode_start_time = time.time()
            pd_binary_gt = []
            pd_binary_pred = []

            if mode == 'train':
                visual_net.train()
                evaluator.train()
                torch.set_grad_enabled(True)
            else:
                visual_net.eval()
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

                if mode == 'train':
                    gt = get_gt(data, config['EVALUATOR']['PREDICT_TYPE'])
                    if config['CRITERION'].get('USE_WEIGHTS'):
                        config['CRITERION']['WEIGHTS'] = get_crossentropy_weights(gt, config['EVALUATOR'])

                    if config['OPTIMIZER'].get('USE_SAM'):
                        for m in [visual_net, evaluator]:
                            enable_running_stats(m)
                        probs = _forward_visual(visual_net, evaluator, data, args)
                        gt_t = gt.float().to(args.device)
                        if probs.dim() == 2 and probs.size(1) == 1:
                            probs = probs.view(-1)
                        loss = criterion(probs, gt_t)
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        for m in [visual_net, evaluator]:
                            disable_running_stats(m)
                        probs2 = _forward_visual(visual_net, evaluator, data, args)
                        if probs2.dim() == 2 and probs2.size(1) == 1:
                            probs2 = probs2.view(-1)
                        compute_loss(criterion, probs2, gt_t, args).backward()
                        optimizer.second_step(zero_grad=True)
                    else:
                        probs = _forward_visual(visual_net, evaluator, data, args)
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
                        print(f'| epoch {epoch:3d} | {mode} | {batch_number:3d}/{n_batches:3d} | LR {lr:7.6f} | '
                              f'ms/batch {ms_per_batch:5.2f} | loss {log_interval_loss / log_interval:8.5f} |')
                        writer.add_scalar(f'Loss_per_{log_interval}_batches/{mode}',
                                          log_interval_loss / log_interval,
                                          epoch * n_batches + batch_number)
                        log_interval_loss = 0
                        batches_start_time = time.time()
                else:
                    with torch.no_grad():
                        probs = _forward_visual(visual_net, evaluator, data, args)
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

            writer.add_scalar(f'Loss_per_epoch/{mode}', average_loss, epoch)
            writer.add_scalar(f'Accuracy/{mode}', accuracy * 100, epoch)

            [[tp, fp], [fn, tn]] = standard_confusion_matrix(pd_binary_gt, pd_binary_pred)
            tpr, tnr, precision, recall, f1_score = get_classification_scores(pd_binary_gt, pd_binary_pred)
            log_and_print(base_logger, f'  CM TP={tp:.0f} FP={fp:.0f} FN={fn:.0f} TN={tn:.0f}  F1={f1_score:.4f}')

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
                    save_model_weights(visual_net, evaluator, epoch, model_type, ckpt_path,
                                       best_metric='val_loss', value=best_val_loss)
                else:
                    stall += 1

            if mode == 'test':
                if accuracy > test_best_acc:
                    test_best_acc = accuracy
                    test_epoch_best_acc = epoch
                    save_model_weights(visual_net, evaluator, epoch, model_type, ckpt_path,
                                       best_metric='acc', value=test_best_acc)
                if f1_score > test_best_f1_score:
                    test_best_f1_score = f1_score
                    test_epoch_best_f1 = epoch
                    best_fold_result = {
                        'fold': fold_idx, 'acc': accuracy, 'tpr': tpr, 'tnr': tnr,
                        'precision': precision, 'recall': recall, 'f1': f1_score, 'epoch': epoch
                    }
                    save_model_weights(visual_net, evaluator, epoch, model_type, ckpt_path,
                                       best_metric='f1', value=test_best_f1_score)

        trace_row['ts'] = datetime.now().isoformat()
        trace_row['fold'] = fold_idx if fold_idx is not None else -1
        trace_row['epoch'] = epoch
        trace_row['lr'] = round(float(scheduler.get_last_lr()[0]), 8)
        trace_row['epoch_sec'] = round(time.time() - epoch_t0, 3)
        if run_trace_dir:
            append_epoch_trace_row(os.path.join(run_trace_dir, 'epoch_metrics.csv'), trace_row)

        scheduler.step()
        if stall >= patience:
            log_and_print(base_logger, f'Early stopping at epoch {epoch} (patience={patience}).')
            break

    return best_fold_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config/config_inference.yaml')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    if str(args.device) == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = YamlConfig(args.config_file)
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    os.makedirs(config['CKPTS_DIR'], exist_ok=True)
    os.makedirs(os.path.join(config['CKPTS_DIR'], config['TYPE']), exist_ok=True)
    os.makedirs(config['MODEL']['WEIGHTS']['PATH'], exist_ok=True)
    config.save(os.path.join(config['OUTPUT_DIR'], config['SAVE_CONFIG_NAME']))
    init_seed(config['MANUAL_SEED'])

    model_type = config['TYPE']
    all_fold_results = []
    n_splits = 5

    for fold_idx in range(n_splits):
        print('=' * 80)
        print(f'Visual-only | Fold {fold_idx + 1}/{n_splits}')
        print('=' * 80)

        fold_output_dir = os.path.join(config['OUTPUT_DIR'], f'fold_{fold_idx + 1}')
        fold_ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'], f'fold_{fold_idx + 1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        os.makedirs(fold_ckpt_path, exist_ok=True)

        base_logger = get_logger(
            os.path.join(fold_output_dir, f'{config["TYPE"]}_fold{fold_idx + 1}.log'),
            f'{config["LOG_TITLE"]} | Fold {fold_idx + 1}'
        )
        writer = SummaryWriter(os.path.join(fold_output_dir, 'runs'))

        dataloaders, fold_info = get_dataloaders(
            config['DATA'], fold_idx=fold_idx, n_splits=n_splits, seed=config['MANUAL_SEED']
        )
        dataloaders['_train_labels'] = fold_info['train_labels']
        log_and_print(base_logger, str(fold_info))

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

        visual_net, evaluator = get_models(config['MODEL'], args, model_type, fold_ckpt_path)
        mcfg = config['MODEL']
        mcfg.setdefault('EARLY_STOP_PATIENCE', 15)

        fold_result = main(
            dataloaders, visual_net, evaluator, base_logger, writer, mcfg, args,
            model_type, fold_ckpt_path, fold_idx=fold_idx + 1,
            run_trace_dir=fold_output_dir,
        )
        all_fold_results.append(fold_result)
        writer.close()
        print(f'Fold {fold_idx + 1} 结果: {fold_result}')

    acc_list = [x['acc'] for x in all_fold_results]
    tpr_list = [x['tpr'] for x in all_fold_results]
    tnr_list = [x['tnr'] for x in all_fold_results]
    f1_list = [x['f1'] for x in all_fold_results]

    def mean_std(x):
        x = np.array(x, dtype=float)
        return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    am, asd = mean_std(acc_list)
    print('\n5-Fold (Visual-only) Summary')
    print(f'Accuracy : {am * 100:.2f} ± {asd * 100:.2f}')
    print(f'TPR      : {mean_std(tpr_list)[0] * 100:.2f} ± {mean_std(tpr_list)[1] * 100:.2f}')
    print(f'TNR      : {mean_std(tnr_list)[0] * 100:.2f} ± {mean_std(tnr_list)[1] * 100:.2f}')
    print(f'F1-score : {mean_std(f1_list)[0]:.4f} ± {mean_std(f1_list)[1]:.4f}')

    results_csv_path = os.path.join(config['OUTPUT_DIR'], 'five_fold_results.csv')
    pd.DataFrame(all_fold_results).to_csv(results_csv_path, index=False)

    write_run_session_summary_json(
        os.path.join(config['OUTPUT_DIR'], 'run_session_summary.json'),
        {
            'finished_at': datetime.now().isoformat(),
            'config_file': os.path.abspath(args.config_file),
            'model_type': config['TYPE'],
            'n_splits': n_splits,
            'mean_acc': am,
            'std_acc': asd,
            'mean_f1': mean_std(f1_list)[0],
            'std_f1': mean_std(f1_list)[1],
            'results_csv': os.path.abspath(results_csv_path),
            'per_fold_epoch_traces': [
                os.path.abspath(os.path.join(config['OUTPUT_DIR'], f'fold_{i + 1}', 'epoch_metrics.csv'))
                for i in range(n_splits)
            ],
        },
    )
    print(f"会话汇总: {os.path.join(config['OUTPUT_DIR'], 'run_session_summary.json')}")
