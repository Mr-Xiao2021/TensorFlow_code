import os
import math
import argparse
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import models
from my_dataset1 import MySegmentationDataset
from utils1 import read_split_data, train_one_epoch, evaluate

def main(args):
        # Parse device string into a list
    device_list = args.device.split(',')
    
    if not all([d.startswith('cuda:') for d in device_list]):
        raise ValueError("All device ids should start with 'cuda:'")

    # Set the first device as the primary device
    device = torch.device(device_list[0] if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_labels_path, val_images_path, val_labels_path = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(256),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    total_start_time = time.time()
    # 实例化训练数据集
    train_dataset = MySegmentationDataset(images_path=train_images_path,
                                          labels_path=train_labels_path,
                                          transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MySegmentationDataset(images_path=val_images_path,
                                        labels_path=val_labels_path,
                                        transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del weights_dict['classifier.4.weight']
        del weights_dict['classifier.4.bias']
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
            else:
                print("Training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Use DataParallel for multi-GPU training
    if len(device_list) > 1:
        model = torch.nn.DataParallel(model, device_ids=[torch.device(d) for d in device_list])
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch)

        scheduler.step()

        val_loss, val_acc = evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    total_end_time = time.time()
    print("Total training time: {:.2f} seconds".format(total_end_time - total_start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=13)  # 增加背景类
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default="/mnt/7T/zhaoxu/data/DriveSeg")
    parser.add_argument('--weights', type=str, default='', help='./vit_base_patch16_224_in21k.pth')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)