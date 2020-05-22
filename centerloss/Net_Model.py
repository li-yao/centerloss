import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),  # 28*28
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),  # 28*28
            nn.BatchNorm2d(32),
            nn.PReLU(),
            # 大小没变，相当于做了特征融合

            nn.MaxPool2d(2, 2),  # 14*14  特征融合充分，取大值就可以
            nn.Conv2d(32, 64, 5, 1, 2),  # 14*14
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),  # 14*14
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # 相当于再做两次特征融合

            nn.MaxPool2d(2, 2),  # 7*7 下采样
            nn.Conv2d(64, 128, 5, 1, 2),  # 7*7
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, 5, 1, 2),  # 7*7
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # 再融合

            nn.MaxPool2d(2, 2)  # 3*3
        )

        self.feature = nn.Linear(128*3*3, 2)  # 为了表示成(x, y)二维，所以输出2
        self.output = nn.Linear(2, 10)  # 10种可能性

    def forward(self, x):
        y_conv = self.conv_layer(x)
        y_conv = torch.reshape(y_conv, [-1, 128*3*3])
        y_feature = self.feature(y_conv)  # [n,2]
        y_output = torch.log_softmax(self.output(y_feature), dim=1)
        return y_feature, y_output  # [n,10]

    def visualize(self, feat, labels, epoch):
        # plt.ion()
        color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
                 '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.clf()  # clear current figure
        for i in range(10):
            plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=color[i])  # 按照标签种类，画出图
            # plt.plot(x,y)
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')  # legend图例 loc：loacation
        # plt.xlim(xmin=-5,xmax=5)
        # plt.ylim(ymin=-5,ymax=5)
        plt.title("epoch=%d" % epoch)  # % 1.求模运算  2.格式化输出  title用于设置图像标题
        plt.savefig('./images/epoch=%d.jpg' % epoch)
        # plt.draw()
        # plt.pause(0.001)
