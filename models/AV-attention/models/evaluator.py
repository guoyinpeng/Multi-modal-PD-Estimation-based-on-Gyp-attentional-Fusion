import torch.nn as nn


class MLP_block(nn.Module):
    def __init__(self, feature_dim):
        super(MLP_block, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class Evaluator(nn.Module):
    def __init__(self, feature_dim, predict_type):
        super(Evaluator, self).__init__()
        assert predict_type in ["pd-binary"]
        self.predict_type = predict_type
        self.evaluator = MLP_block(feature_dim)

    def forward(self, feats_avg):
        return self.evaluator(feats_avg)


if __name__ == "__main__":
    ev = Evaluator(256, predict_type="pd-binary")
    print(*ev.parameters())
