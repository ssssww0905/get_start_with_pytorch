import os
from shutil import copyfile, rmtree
import random


def mk_dir(dir):
    if os.path.exists(dir):
        rmtree(dir)
    os.makedirs(dir)


def main():
    random.seed(0)
    split_rate = 0.1

    cwd = os.getcwd()
    data_path = os.path.join(cwd, "flower_data")
    origin_data_path = "../dataset/flower_dataset"
    assert os.path.exists(origin_data_path), \
        "path {} does not exist.".format(origin_data_path)

    # os.listdir(path)      返回指定路径下的文件和文件夹列表
    # os.path.isdir(path)	判断路径是否为目录
    flower_class = [cla for cla in os.listdir(origin_data_path)
                    if os.path.isdir(os.path.join(origin_data_path, cla))]

    train_path = os.path.join(data_path, "train")
    mk_dir(train_path)
    for cla in flower_class:
        mk_dir(os.path.join(train_path, cla))

    valid_path = os.path.join(data_path, "valid")
    mk_dir(valid_path)
    for cla in flower_class:
        mk_dir(os.path.join(valid_path, cla))

    for cla in flower_class:
        valid_num, train_num = 0, 0
        cla_path = os.path.join(origin_data_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        valid_index = random.sample(range(num), k=int(num * split_rate))

        for index, image in enumerate(images):
            image_path = os.path.join(cla_path, image)
            if index in valid_index:
                copyfile(image_path, os.path.join(valid_path, cla, "{}.jpg".format(valid_num)))
                valid_num += 1
            else:
                copyfile(image_path, os.path.join(train_path, cla, "{}.jpg".format(train_num)))
                train_num += 1


if __name__ == '__main__':
    print("data prepare start!")
    main()
    print("data prepare done!")

