import torch


class Predict10(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU6(),
            torch.nn.Linear(64, 10),
            torch.nn.ReLU6(),
        )

    def forward(self, x):
        return self.decoder(x)

