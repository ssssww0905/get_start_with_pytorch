import os
import sys
import json
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from model import VGGNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data_loader(batch_size):
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
    train_path = "../alexnet/flower_data/train"
    valid_path = "../alexnet/flower_data/valid"
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
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)
    return train_loader, valid_loader


def train(data_loader, model, loss_fn, optimizer, epoch, EPOCH, writer):
    model.train()
    train_loss = 0.
    desc = "[EPOCH {:>3d} / {:>3d}] TRAIN".format(epoch+1, EPOCH)
    with tqdm(data_loader, desc=desc, ncols=80, file=sys.stdout) as train_bar:
        for (x, y) in train_bar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    print("[EPOCH {:>3d} / {:>3d}] TRAIN LOSS : {:.6f} ".format(epoch + 1, EPOCH, train_loss))
    writer.add_scalar('Train Loss', train_loss, epoch + 1)


def valid(data_loader, model, loss_fn, epoch, EPOCH, writer):
    model.eval()
    valid_loss = 0.
    valid_num = len(data_loader.dataset)
    correct_num = 0.
    desc = "[EPOCH {:>3d} / {:>3d}] VALID".format(epoch + 1, EPOCH)
    with torch.no_grad():
        with tqdm(data_loader, desc=desc, ncols=80, file=sys.stdout) as valid_bar:
            for (x, y) in valid_bar:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)

                valid_loss += loss.item()
                correct_num += (pred.argmax(1) == y).type(torch.float).sum().item()

        print("[EPOCH {:>3d} / {:>3d}] VALID LOSS : {:.6f} ".format(epoch + 1, EPOCH, valid_loss))
        print("[EPOCH {:>3d} / {:>3d}] VALID ACCU : {:.6f} ".format(epoch + 1, EPOCH, correct_num / valid_num))
        writer.add_scalar('Valid Loss', valid_loss, epoch + 1)
        writer.add_scalar('Valid Accu', correct_num / valid_num, epoch + 1)


if __name__ == "__main__":
    writer = SummaryWriter('log')
    vgg_model = "vgg_16"
    batch_size = 32
    lr = 0.0002
    train_loader, valid_loader = prepare_data_loader(batch_size=batch_size)
    model = VGGNet(vgg_model, class_num=5, init_weights=True).to(device)
    # model.load_state_dict(torch.load("model_{}.pth".format(vgg_model))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    EPOCH = 10
    for epoch in range(EPOCH):
        train(train_loader, model, loss_fn, optimizer, epoch, EPOCH, writer)
        valid(valid_loader, model, loss_fn, epoch, EPOCH, writer)

    print("Done!")
    torch.save(model.state_dict(), "model_{}.pth".format(vgg_model))
    writer.close()
