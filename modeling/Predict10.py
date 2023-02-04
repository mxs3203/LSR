import torch


class Predict10(torch.nn.Module):
    def __init__(self, curve_size):
        super().__init__()

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(curve_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.decoder(x)

