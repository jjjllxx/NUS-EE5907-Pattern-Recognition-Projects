import time
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import scipy.io as sio

import torch
import torch.optim as optim

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print(device)

train_set = sio.loadmat("trainingset.mat")
test_set = sio.loadmat("testingset.mat")
x_train = train_set["trainingset"]
x_test = test_set["testingset"]
train_size = len(x_train)
test_size = len(x_test)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = [(i // 119) for i in range(train_size)]
y_test = [(i // 51) for i in range(test_size)]
x_train = x_train.reshape(train_size, 1, 32, 32)
x_test = x_test.reshape(test_size, 1, 32, 32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class ConvNet(torch.nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(20, 50, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(50 * 8 * 8, 50 * 8 * 8),
            torch.nn.ReLU(),
            torch.nn.Linear(50 * 8 * 8, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 26)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 3200)
        x = self.dense(x)
        return x


myModel = ConvNet()
myModel.to(device)

epochs = 50
batch_size = 64
learning_rate = 0.05

optimizer = optim.SGD(myModel.parameters(), lr=learning_rate, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

train_set = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

data_loaders = {"train": train_loader, "test": test_loader}
dataset_sizes = {"train": train_size, "test": test_size}
test_error_rate = []
train_error_rate = []
start = time.time()
myModel.train()
# train part
for epoch in range(epochs):
    train_loss = 0
    train_error = 0
    for data in train_loader:
        # Load inputs and labels and deploy to running device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # Set the gradients to zeros
        optimizer.zero_grad()
        # Forward the batch data through the net
        outputs = myModel(inputs)
        # Compute the average of the losses of the data points in the minibatch
        loss = criterion(outputs, labels)
        # Backward pass to compute gradients
        loss.backward()
        # Do one step of stochastic gradient descent
        optimizer.step()
        # Add the loss of this batch to the running loss
        train_loss += loss.item()
        # Compute the error made on this batch and add it to the running error
        _, predicted = torch.max(outputs.data, dim=1)
        train_error += (predicted != labels).sum().item()
    # Compute stats for the full training set
    total_loss = train_loss / train_size
    total_error = train_error / train_size
    train_error_rate.append(total_error)
    elapsed = (time.time() - start) / 60
    print('epoch= {} \t time= {:.2f} min \t loss= {:.3f} \t error= {:.2f}%'.format(epoch, elapsed, total_loss,
                                                                                   total_error * 100))
    # evaluate on test set
    myModel.eval()
    test_error = 0
    for data in test_loader:
        # Load inputs and labels and deploy to running device
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # Forward batch data through the net
        outputs = myModel(images)
        _, predicted = torch.max(outputs.data, dim=1)
        # Compute the error made on this batch and add it to the running error
        test_error += (predicted != labels).sum().item()
    total_error = test_error / test_size
    test_error_rate.append(total_error)
    print('error rate on test set = {:.2f}%'.format(total_error * 100))
    myModel.train()

epo = [i + 1 for i in range(epochs)]
# draw accuracy figure
plt.figure(1)
plt.plot(epo, train_error_rate, label='Train error')
plt.plot(epo, test_error_rate, label='Test error')
plt.xlabel('Epoch times')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title("Training and testing accuracy")
plt.grid()
plt.show()
