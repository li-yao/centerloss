import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from centerloss.Net_Model import Net
from centerloss.loss import centerloss_net
import os
import numpy as np


if __name__ == '__main__':

    save_path_net = "models/net.pth"
    save_path_centerloss_net = "models/centerloss_net.pth"
    train_data = torchvision.datasets.MNIST(root="./MNIST", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.5, ], std=[0.5, ])]))
    train_loader = data.DataLoader(dataset=train_data, shuffle=True, batch_size=100, num_workers=4)
    # num_workers是加载数据的线程数目

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    centerloss_net = centerloss_net().to(device)

    if os.path.exists(save_path_net):
        net.load_state_dict(torch.load(save_path_net))
    else:
        print("No Param")

    if os.path.exists(save_path_centerloss_net):
        net.load_state_dict(torch.load(save_path_net))
    else:
        print("No Param")

    'CrossEntropyLoss()=torch.log(torch.softmax(None))+nn.NLLLoss()'
    'CrossEntropyLoss()=log_softmax() + NLLLoss() '
    'nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合'
    '多分类交叉熵CrossEntropyLoss与NLLLoss意义相同，唯一不同的只是一个有log+softmax,一个没有，意义是一样的'
    # lossfn_cls = nn.CrossEntropyLoss()
    lossfn_cls = nn.NLLLoss()
    optimzer1 = torch.optim.Adam(net.parameters())  # 1.先Adam 再SGD 2.SGD加动量训练
    # optimzer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    optimzer2_centerloss = torch.optim.SGD(centerloss_net.parameters(), lr=1e-3)
    # optimzer = torch.optim.SGD(net.parameters(),lr=1e-3, momentum=0.9, weight_decay=0.0005)

    epoch = 0
    while True:
        feat_loader = []
        label_loader = []
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            feature, output = net.forward(x)
            # print(feature.shape)#[N,2]
            # print(output.shape)#[N,10]
            # center = nn.Parameter(torch.randn(output.shape[1], feature.shape[1]))
            # print(center.shape)#[10,2]

            loss_cls = lossfn_cls(output, y)  # 这里的output在网络里已经进行过Logsoftmax运算了，所以直接进行NLLLoss损失计算
            y = y.float()

            loss_center = centerloss_net(feature, y, 2)  # 2为lambdas权重

            loss = loss_cls+loss_center

            optimzer1.zero_grad()
            optimzer2_centerloss.zero_grad()
            loss.backward()
            optimzer1.step()
            optimzer2_centerloss.step()

            # feature.shape=[100,2]
            # y.shape=[100]
            feat_loader.append(feature)
            label_loader.append(y)

            if i % 600 == 0:
                print("epoch:", epoch, "i:", i, "total:", loss.item(), "softmax_loss:", loss_cls.item(), "center_loss:", loss_center.item())

        feat = torch.cat(feat_loader, 0)  # Concatenates the given sequence of seq tensors in the given dimension.
        labels = torch.cat(label_loader, 0)
        '---------------'
        # print(feat.shape)#feat.shape=[60000,2]
        # print(labels.shape)#label.shape=[60000]
        '-------------------'
        net.visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
        epoch += 1
        torch.save(net.state_dict(), save_path_net)
        torch.save(centerloss_net.state_dict(), save_path_centerloss_net)
        if epoch == 150:
            break
