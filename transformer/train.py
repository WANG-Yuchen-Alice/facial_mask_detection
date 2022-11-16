from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate

from torch.utils.tensorboard import SummaryWriter

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # Create train dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # Create val dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=1,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    # print(model)
    model.head = torch.nn.Sequential(torch.nn.Linear(768, 512, bias=True), torch.nn.ReLU(),torch.nn.Linear(512, args.num_classes, bias=True)).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # Delete useless weight
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['head.weight', 'head.bias']  # 'pre_logits.fc.weight', 'pre_logits.fc.bias',
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # Freeze the weight except head, pre_logits
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)



    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []


    writer = SummaryWriter("runs/logs_fina")  # Direction to save log(for tensorboard)
    for epoch in range(args.epochs):
        # train

        if epoch == 1:
            train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch, writer=writer)

        scheduler.step()

        # validate
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





        writer.add_scalar('train/loss', train_loss, epoch)  # To draw loss, with epoch as the x axis
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.close()



        epoch_num = range(epoch+1)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)


        plt.subplot(2, 1, 1)
        plt.plot(epoch_num, train_acc_list, 'o-', label='train_acc')
        plt.plot(epoch_num, val_acc_list, 'o-', label='val_acc')
        plt.legend()
        plt.title('Train/Val accuracy vs epoches')
        x = MultipleLocator(1)  # X axis, 1 as a scale
        ax = plt.gca()
        ax.xaxis.set_major_locator(x)
        plt.ylabel('Accuracy')
        plt.subplots_adjust(wspace=0, hspace=0.3)
        plt.subplot(2, 1, 2)
        plt.plot(epoch_num, train_loss_list, 'o-', label='train_loss')
        plt.plot(epoch_num, val_loss_list, 'o-', label='val_loss')
        plt.legend()
        plt.title('Train/Val loss vs epoches')
        x = MultipleLocator(1)  # X axis, 1 as a scale
        ax = plt.gca()
        ax.xaxis.set_major_locator(x)
        plt.ylabel('Loss')
        plt.savefig("./{}.jpg".format(epoch))
        plt.show()


        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)

    # The location of the dataset
    parser.add_argument('--data-path', type=str,
                        default=r"C:\temp_can\flower_photos")
    parser.add_argument('--model-name', default='vit_base_patch16_224', help='create model name')

    # pretrain weight dir
    parser.add_argument('--weights', type=str, default=r'C:\temp_can\vit_base_patch16_224.pth',
                        help='initial weights path')
    # Whether to freeze the weight
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
