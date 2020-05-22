import torch


def center_loss():
    data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32)  # 5,2
    label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)
    center = torch.tensor([[1, 1], [2, 2]], dtype=torch.float32)

    center_exp = center.index_select(dim=0, index=label.long())
    # print(data)
    print(center_exp)

    count = torch.histc(label, bins=int(max(label).item() + 1), min=int(min(label).item()), max=int(max(label).item()))
    # tensor([3., 2.])  --统计不重复的类别出现的次数 bins为类别  知道每一个类有多少个，才能算每个类的损失
    # The elements are sorted into equal width bins between min and max.
    # If min and max are both zero, the minimum and maximum values of the data are used.
    # print(count)

    count_exp = count.index_select(dim=0, index=label.long())
    # print(count_exp)  # tensor([3., 3., 2., 3., 2.]) (5,1)

    # center loss
    # loss = torch.pow(data - center_exp, 2)  # pow 指数
    # loss = torch.sum(torch.pow(data - center_exp, 2), dim=1)
    # loss = torch.div(torch.sum(torch.pow(data - center_exp, 2), dim=1), count_exp)
    # tensor([ 4.3333, 13.6667, 30.5000, 37.6667, 12.5000])
    loss = torch.mean(torch.div(torch.sum(torch.pow(data - center_exp, 2), dim=1), count_exp))

    # print(loss)
    return loss


if __name__ == '__main__':
    center_loss = center_loss()
    print(center_loss)
