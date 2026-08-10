import torch.nn as nn


class ImitationLearningNet(nn.Module):
    """Architecture kept identical to the original Keras model (Dense 64 -> Dense 16 ->
    Dense num_warehouses, ReLU/ReLU), shared between training (train_imitation_learning.ipynb)
    and inference (ImitationLearningPolicy) so the two never drift apart. Outputs raw
    logits, not softmax probabilities - argmax over logits and argmax over softmax(logits)
    agree, and CrossEntropyLoss expects raw logits during training."""

    def __init__(self, num_warehouses):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * num_warehouses, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, num_warehouses),
        )

    def forward(self, x):
        return self.net(x)
