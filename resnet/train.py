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
from model import ResNet, resnet34, resnet50
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data_loader(batch_size):
    # set transforms
    transforms_ = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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


def valid(data_loader, model, loss_fn, epoch, EPOCH, writer, best_acc):
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

        valid_acc = correct_num / valid_num

        print("[EPOCH {:>3d} / {:>3d}] VALID LOSS : {:.6f} ".format(epoch + 1, EPOCH, valid_loss))
        print("[EPOCH {:>3d} / {:>3d}] VALID ACCU : {:.6f} ".format(epoch + 1, EPOCH, valid_acc))

        writer.add_scalar('Valid Loss', valid_loss, epoch + 1)
        writer.add_scalar('Valid Accu', valid_acc, epoch + 1)

        if valid_acc > best_acc[0]:
            best_acc[0] = valid_acc
            torch.save(model.state_dict(), "{}_trained.pth".format(model.name))


if __name__ == "__main__":
    batch_size = 32
    lr = 0.0001
    train_loader, valid_loader = prepare_data_loader(batch_size=batch_size)

    # model = resnet50(class_num=1000, include_top=True)
    # model.load_state_dict(torch.load("{}_pretrained.pth".format(model.name), map_location=device), strict=False)
    # # change fc layer to match this classification
    # model.fc = nn.Linear(model.fc.in_features, 5)

    model = resnet34(class_num=5, include_top=True)
    model.load_state_dict(torch.load("{}_trained.pth".format(model.name), map_location=device), strict=False)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    EPOCH = 3
    writer = SummaryWriter("log_{}".format(model.name))
    best_acc = [0.]
    for epoch in range(EPOCH):
        train(train_loader, model, loss_fn, optimizer, epoch, EPOCH, writer)
        valid(valid_loader, model, loss_fn, epoch, EPOCH, writer, best_acc)

    print("Done!")
    writer.close()
