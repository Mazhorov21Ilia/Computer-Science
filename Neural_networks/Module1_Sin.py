%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)
"""
Библиотека для наглядности работы нейросети
"""

import torch

x_train = torch.rand(100)
x_train = x_train * 20.0 - 10.0
"""
Задаем значение начального тензоров с помощью функции рандом 
и приближаем их к более приятным значениям
"""


y_train = torch.sin(x_train)
noise = torch.rand(100) - 0.5 
y_train = y_train + noise
"""
Считаем синус от всех рандомных значений и накладываем шум, 
чтобы нейросеть "научилась", а не "зубрила"
"""


x_train.unsqueeze_(1)
y_train.unsqueeze_(1)
"""
Переводим строчки в столбцы
"""


x_validation = torch.linspace(-10, 10, 100)
y_validation = torch.sin(x_validation.data)
"""
Задаем 100 рандомных значений для тензора x в промежутке [-10, 10]
"""


x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

class SineNet(torch.nn.Module):
  def __init__(self, n_hidden_neurons):
    super(SineNet, self).__init__()
    self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
    self.act1 = torch.nn.Sigmoid()
    self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act1(x)
    x = self.fc2(x)
    return x
"""
Класс самой нейросети, в которой мы создаем три слоя, только
один из которых скрытый, указываем количество нейронов(n_hidden_neurons)
и задаем функцию активации.
С помощью функции forward мы как раз будем изменять наши данные
"""


sine_net = SineNet(50)
"""
Задаем классу SineNet 50 нейронов(с потолка)
"""


optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.01)
"""
Наш оптимайзер Адам, будет учить нейросеть
по средством градиентного спуска
"""


def loss(pred, target):
  squares = (pred - target) ** 2
  return squares.mean()
"""
Функция потерь(квадратичная ошибка).
Показывает на сколько выходные значения нейросети
отличаются от желаемых
"""


for epoch_index in range(2000):
  y_pred = sine_net.forward(x_train)
  loss_val = loss(y_pred, y_train)
  
  loss_val.backward()

  optimizer.step()
  optimizer.zero_grad()
"""
Цикл обучения Нейросети.
Для начала проводим наши тренировочные значения x по нашей функции активации.
Далее считаем количество потерь и выполняем обратное распространение ошибки.
В конце делаем градиентный спуск и обнуляем наши градиенты
"""

  
y_pred = sine_net.forward(x_validation)
plt.plot(x_validation.numpy(), y_validation.numpy(), 'o')
plt.plot(x_validation.numpy(), y_pred.data.numpy(), 'o', c='r')
