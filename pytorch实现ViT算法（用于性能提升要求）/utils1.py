import os
import sys
import json
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "Dataset root: {} does not exist.".format(root)

    frame_dir = os.path.join(root, "frames")
    label_dir = os.path.join(root, "labels")
    assert os.path.exists(frame_dir), "Frames folder {} does not exist.".format(frame_dir)
    assert os.path.exists(label_dir), "Labels folder {} does not exist.".format(label_dir)

    # Load config.json
    config_path = os.path.join(root, "config.json")
    assert os.path.exists(config_path), "Config file {} does not exist.".format(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get all image file names
    all_images = [f for f in os.listdir(frame_dir) if f.endswith('_img.jpg')]
    all_images.sort()

    # Shuffle and split into training and validation sets
    num_val = int(len(all_images) * val_rate)
    val_images = random.sample(all_images, k=num_val)
    train_images = [img for img in all_images if img not in val_images]

    train_img_paths = [os.path.join(frame_dir, img) for img in train_images]
    train_lbl_paths = [os.path.join(label_dir, img.replace('_img.jpg', '_gt_id.png')) for img in train_images]

    val_img_paths = [os.path.join(frame_dir, img) for img in val_images]
    val_lbl_paths = [os.path.join(label_dir, img.replace('_img.jpg', '_gt_id.png')) for img in val_images]

    return train_img_paths, train_lbl_paths, val_img_paths, val_lbl_paths


def plot_data_loader_image(data_loader, class_indices):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            label = labels[i].cpu().numpy()

            plt.subplot(2, plot_num, i+1)
            plt.imshow(img.astype('uint8'))
            plt.subplot(2, plot_num, i+1+plot_num)
            plt.imshow(label, alpha=0.5, cmap='jet')  # Overlay label map on the image
            plt.xticks([])  
            plt.yticks([])  
        plt.show()


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_correct = torch.zeros(1).to(device)
    sample_num = 0
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        output = model(images)['out']  # 预测结果
        output = torch.nn.functional.interpolate(output, size=labels.shape[1:], mode='bilinear', align_corners=True)  # 上采样

        loss = loss_function(output, labels)
        loss.backward()
        accu_loss += loss.detach()

        optimizer.step()
        optimizer.zero_grad()
         # 计算像素准确率
        preds = torch.argmax(output, dim=1)
        accu_correct += torch.sum(preds == labels)
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, accu_loss.item() / (step + 1),accu_correct.item() / (sample_num * labels.size(1) * labels.size(2)))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_correct.item() / (sample_num * labels.size(1) * labels.size(2))

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_correct = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]
        output = model(images)['out']  # 预测结果
        output = torch.nn.functional.interpolate(output, size=labels.shape[1:], mode='bilinear', align_corners=True)  # 上采样

        loss = loss_function(output, labels)
        accu_loss += loss
        # 计算像素准确率
        preds = torch.argmax(output, dim=1)
        accu_correct += torch.sum(preds == labels)

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, accu_loss.item() / (step + 1),accu_correct.item() / (sample_num * labels.size(1) * labels.size(2)))

    return accu_loss.item() / (step + 1), accu_correct.item() / (sample_num * labels.size(1) * labels.size(2))