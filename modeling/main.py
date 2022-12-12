from torch.autograd import Variable
from torch.utils.data import DataLoader

from modeling.Curve_Dataloader import Curve_Loader
import torch
import numpy as np
from modeling.Predict10 import Predict10
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))

loader = Curve_Loader("~/LSR/modeling/input_data_with_fft.csv", fft_size=9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = 'cpu'

train_size = int(len(loader) * 0.75)
test_size = len(loader) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(loader, [train_size, test_size])

model = Predict10(curve_size=9)
model.to(device)
batch_size = 128
epochs = 2000
LR = 1e-3
WD = 1e-3
cost_func = torch.nn.MSELoss(reduction="mean", reduce=True)
optimizer = torch.optim.Adagrad(model.parameters(),lr=LR, weight_decay=WD)

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size,num_workers=10, shuffle=True)


def makeCurve(curve, real_ten_nums, predicted_ten_nums):
    real_ten_nums = [int(10**(item-0.0001)) for item in real_ten_nums]
    predicted_ten_nums = [int(10**(item-0.0001)) for item in predicted_ten_nums]
    plt.plot(range(0, 9, 1), curve)
    plt.text(0, 1, real_ten_nums, fontsize=8, c="green")
    plt.text(0, 0.80, predicted_ten_nums, fontsize=8, c="red")
    plt.show()
    pass

best_model = None
best_loss = float("+Inf")
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
    if(np.mean(test_ep_loss) < best_loss):
        best_loss = np.mean(test_ep_loss)
        best_model = model
        print("Best loss detected, saving model as the best model...")
        torch.save(model.state_dict(), "best_model.pth")