import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
import numpy as np
from model import LeNet
from imgshow import imshow
device = 'cuda' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# show image for fun
# dataiter = iter(trainloader)
# print(len(dataiter))
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10000 == 0:
            print("Epoch: {:>4d} Train Loss:{:>.6f} [{:>6d} / {:>6d}]".format(epoch, loss.item(), batch * len(X), size))


def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            _, pred_id = torch.max(pred.data, 1)
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()

    print("Epoch: {:>4d}  Test Accuracy :{:>6d} / {:>6d}".format(epoch, correct, size))

model = LeNet().to(device)
# model.load_state_dict(torch.load("model.pth"))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
for epoch in range(10):
    train(trainloader, model, loss_fn, optimizer, epoch)
    test(testloader, model, loss_fn, epoch)

print("Done!")
torch.save(model.state_dict(), "model.pth")
