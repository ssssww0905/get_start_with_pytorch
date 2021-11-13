import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import ResNet, resnet34, resnet50
from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # preprocessing image
    img_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    img_path = "test.jpg"
    img = Image.open(img_path)
    plt.imshow(img)
    img = img_transforms(img)
    img = torch.unsqueeze(img, dim=0)
    # load index2class dict
    with open('index2class.json', 'r') as json_file:
        index2class = json.load(json_file)

    model = resnet34(class_num=5, include_top=True)
    model.load_state_dict(torch.load("resnet34_trained.pth", map_location=device))
    model.to(device)

    # predict
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        pred = torch.squeeze(model(img))

        pred_prob = torch.softmax(pred, dim=0).cpu()
        pred_index = torch.argmax(pred_prob).numpy()

    plt.title("class : {} prob : {}".format(index2class[str(pred_index)], pred_prob.numpy()[pred_index]))
    plt.savefig("test_pred.pdf")
    plt.show()


if __name__ == "__main__":
    main()
