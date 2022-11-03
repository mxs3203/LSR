from torch.autograd import Variable
from torch.utils.data import DataLoader

from modeling.Curve_Dataloader import Curve_Loader
import torch
import numpy as np
from modeling.Predict10 import Predict10
import matplotlib.pyplot as plt

loader = Curve_Loader("~/LSR/modeling/input_data.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
train_size = int(len(loader) * 0.75)
test_size = len(loader) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(loader, [train_size, test_size])

model = Predict10(curve_size=322)
model.to(device)
batch_size = 128
epochs = 1000
LR = 1e-3
WD = 1e-5
cost_func = torch.nn.MSELoss()
optimizer = torch.optim.Adagrad(model.parameters(),lr=LR, weight_decay=WD)

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size,num_workers=10, shuffle=True)


def makeCurve(curve, real_ten_nums, predicted_ten_nums):
    real_ten_nums = [round(item, 2) for item in real_ten_nums]
    predicted_ten_nums = [round(item, 2) for item in predicted_ten_nums]
    plt.plot(range(0, 322, 1), curve)
    plt.text(0, 0, real_ten_nums, fontsize=8)
    plt.text(0, 0.1, predicted_ten_nums, fontsize=8)
    plt.show()
    pass


for ep in range(epochs):
    train_ep_loss,test_ep_loss = [],[]
    model.train()
    for x, y in trainLoader:
        cost_func.zero_grad()
        y_hat = model(x.to(device).float())
        loss = cost_func(y_hat, y.float())
        train_ep_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        model.eval()
        for x, y in valLoader:
            y_hat = model(x.to(device).float())
            loss = cost_func(y_hat, y.float())
            test_ep_loss.append(loss.item())
    if ep % 10 == 0:
        makeCurve(x[0, :].detach().cpu().numpy(), y[0, :].detach().cpu().numpy(), y_hat[0, :].detach().cpu().numpy())
    print("Epoch: {} \n\t Train Loss: {} \n\t Test Loss: {}".format(ep, np.mean(train_ep_loss),np.mean(test_ep_loss)))
