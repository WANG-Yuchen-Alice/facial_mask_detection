import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    # random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Walk through the folders, one for each category
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # Sort to ensure that the order of all platforms is consistent
    flower_class.sort()
    # Generate the category name and the corresponding numeric index
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # Stores all picture paths for the training set
    train_images_label = []  # Store training set image corresponding index information
    val_images_path = []  # Store all picture paths for the validation set
    val_images_label = []  # Store the index information of the image corresponding to the verification set
    every_class_num = []  # Store the total number of samples for each category
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # Supported file suffixes
    # Traverse the files under each folder
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # All supported file paths are traversed
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # Sort to ensure that the order of all platforms is consistent
        images.sort()
        # Gets the index corresponding to the category
        image_class = class_indices[cla]
        # Record the number of samples for that category
        every_class_num.append(len(images))
        # Verify samples by proportional random sampling
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # If the path is in the sampled validation set sample, the validation set is stored
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # Otherwise save the training set
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label



def train_one_epoch(model, optimizer, data_loader, device, epoch, writer):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss
    accu_num = torch.zeros(1).to(device)   # Accumulated correct sample number
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        images = images.to(device)
        writer.add_graph(model, images)

        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
