import torch
import torch.nn as nn


class centerloss_net(nn.Module):
    def __init__(self):
        super(centerloss_net, self).__init__()
        self.center = nn.Parameter(torch.randn(10, 2), requires_grad=True)

    def forward(self, feature, label, lambdas):
        # label = label.squeeze()
        center_exp = self.center.index_select(dim=0, index=label.long())
        count = torch.histc(label, bins=int(max(label).item() + 1), min=0, max=int(max(label).item()))
        count_exp = count.index_select(dim=0, index=label.long())  # 按照label的排列生成每个标签对应的个数数字组成的tensor
        center_loss = lambdas / 2 * torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2), dim=1), count_exp))
        return center_loss


if __name__ == '__main__':
    feature = torch.randn((10,2),dtype=float).cuda()
    label = torch.tensor([0,1,1,0,0,1,0,1,0,1],dtype=float).cuda()
    lambdas = 2
    #
    # centerloss_net()(feature, label, lambdas)
    # print(feature)

    # data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32).cuda()
    # label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32).cuda()
    # lambdas = 2
    print(centerloss_net()(feature, label, lambdas))