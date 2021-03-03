import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
#from torchviz import make_dot

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Программа считает коэффициенты для трехчлена a+bx+cx^2
_a = 1
_b = 2
_c = 3

# Генерируем данные для дальнейшего обучения
# Используем функцию random и считаем функцию y, вычисляя результат многочлена и прибавляя Гауссовский шум
np.random.seed(42)
x = np.random.rand(400, 1)
y = _a + _b * x + _c * x**2 +.1 * np.random.randn(400, 1)

torch.manual_seed(42)
# Переводим массивы numpy в тензоры pyTorch
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Оборачиваем тензоры в набор данных и делим данные на обучающее и проверочное множества
dataset = TensorDataset(x_tensor, y_tensor)
train_dataset, val_dataset = random_split(dataset, [320, 80])

# Разбиваем данные на мини-пакеты
train_loader = DataLoader(dataset=train_dataset, batch_size=20)
val_loader = DataLoader(dataset=val_dataset, batch_size=80)

# Создаем модель для нашего многочлена
class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.c = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.a + self.b * x + self.c * x**2

model = _model().to(device)
#print(model.state_dict())

# Устанавливаем скорость обучения и число эпох
lr = 1e-2
n_epochs = 500

# Определяем функцию потерь и оптимизатор
#loss_fn = nn.MSELoss(reduction='mean')
#optimizer = optim.SGD(model.parameters(), lr=lr)

losses = []
val_losses = []

a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
c = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
print(a,b,c)

# функция, выполняющая шаг в цикле тренировки модели
def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

#train_step = make_train_step(model, loss_fn, optimizer)
# Обучаем модель, считаем потери при тренировке и при проверке
for epoch in range(n_epochs):
  
    losses = []
    val_losses = []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        yhat = a + b * x_batch + c * x_batch**2
        error = y_batch - yhat
        loss = (sum(error**2))**(1/2)
        loss.backward()
        with torch.no_grad():
            a -= lr*a.grad
            b -= lr*b.grad
            c -= lr*c.grad
        a.grad.zero_()
        b.grad.zero_()
        c.grad.zero_()
        #loss = train_step(x_batch, y_batch)
        losses.append(loss.item())

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            #model.eval()
            #yhat = model(x_val)
            #val_loss = loss_fn(y_val, yhat)
            yhat = a + b * x_val + c * x_val ** 2
            error = y_val - yhat
            val_loss = (sum(error ** 2)) ** (1 / 2)
            val_losses.append(val_loss.item())

    print("Epoch:" + str(epoch) + " Training loss: " + str(np.mean(losses)) + " Val loss: + "+ str(np.mean(val_losses)))
    
#print(model.state_dict())
print(a)
print(b)
print(c)
print(np.mean(losses))
print(np.mean(val_losses))
