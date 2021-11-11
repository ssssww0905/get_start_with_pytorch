import os
import json
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
import numpy as np
from model import AlexNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def data_loader_prepare(batch_size):
    # set transforms
    transforms_ = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    }

    # data set
    data_path = os.path.join(os.getcwd(), "flower_data")
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    train_dataset = datasets.ImageFolder(root=train_path, transform=transforms_["train"])
    valid_dataset = datasets.ImageFolder(root=valid_path, transform=transforms_["valid"])

    # class dict
    class2index_dict = train_dataset.class_to_idx
    index2class_dict = dict((value, key) for key, value in class2index_dict.items())
    with open('index2class.json', 'w') as json_file:
        json.dump(index2class_dict, json_file)  # test : index2class_dict = json.load(json_file)

    # data loader
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    valid_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)
    return train_loader, valid_loader


def train(data_loader, model, loss_fn, optimizer, epoch, EPOCH):
    train_loss = 0.
    # train_bar = tqdm(data_loader, ncols=80)
    model.train()
    for _, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # train_bar.desc = "[EPOCH {:>3d} / {:>3d}] TRAIN LOSS : {:.6f} ".format(epoch+1, EPOCH, train_loss)

    print("[EPOCH {:>3d} / {:>3d}] TRAIN LOSS : {:.6f} ".format(epoch + 1, EPOCH, train_loss))


def valid(data_loader, model, loss_fn, epoch, EPOCH):
    valid_loss = 0.
    # valid_bar = tqdm(data_loader, ncols=80)
    valid_num = len(data_loader.dataset)
    correct_num = 0
    model.eval()
    with torch.no_grad():
        for _, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            valid_loss += loss.item()
            # valid_bar.desc = "[EPOCH {:>6d} / {:>6d}] VALID LOSS : {:.6f} ".format(epoch + 1, EPOCH, valid_loss)
            correct_num += (pred.argmax(1) == y).type(torch.int).sum().item()

        print("[EPOCH {:>3d} / {:>3d}] VALID LOSS : {:.6f} ".format(epoch + 1, EPOCH, valid_loss))
        print("[EPOCH {:>3d} / {:>3d}] VALID ACRR : {:.6f} ".format(epoch + 1, EPOCH, float(correct_num) / valid_num))

if __name__ == "__main__":
    batch_size = 32
    lr = 0.0002
    train_loader, valid_loader = data_loader_prepare(batch_size=batch_size)
    model = AlexNet(class_num=5, init_weights=True).to(device)
    model.load_state_dict(torch.load("model.pth"))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    EPOCH = 10
    for epoch in range(EPOCH):
        train(train_loader, model, loss_fn, optimizer, epoch, EPOCH)
        valid(valid_loader, model, loss_fn, epoch, EPOCH)

    print("Done!")
    torch.save(model.state_dict(), "model.pth")

