# main.py
import torch
import torch.nn as nn
import time
import json
from models import SlowFastGCN
from NTURGBD_Datasets import Skeleton_Datasets


def train(ep):
    model.train()
    total = 0
    top1_correct = 0
    top5_correct = 0
    train_loss = 0
    show_batch = 10
    ss = time.time()
    for ix, sample in enumerate(trainloader):
        opt.zero_grad()
        label = sample["label"].cuda()
        data_fast = sample["fast"]["data"][:, :c, :, :, :]
        data_slow = sample["slow"]["data"][:, :c, :, :, :]
        data_fast = data_fast.cuda()
        data_slow = data_slow.cuda()
        output = model(x_fast=data_fast, x_slow=data_slow)

        loss = criterion(output, label)
        loss.backward()
        opt.step()  # I forget this, fuck...
        train_loss += loss.item()
        _, indices = torch.sort(output.detach(), descending=True)
        top1_correct += indices[:, 0].eq(label).sum().item()
        top5_correct += indices[:, :5].eq(label.view(-1, 1).repeat(1, 5)).sum().item()
        total += label.size(0)

        if ((ix + 1) % show_batch) == 0:
            ee = time.time()
            t = ee - ss
            h = t // 3600
            m = (t % 3600) // 60
            s = t % 60
            et = t * len(trainloader) // show_batch
            eh = et // 3600
            em = (et % 3600) // 60
            es = et % 60
            ss = time.time()
            print("epoch: {}, batch: {}/{} time: {:2d}h|{:2d}m|{:2d}s".format(ep, ix + 1, len(trainloader), int(h), int(m), int(s)))
            print("time(epoch): {:2d}h|{:2d}m|{:2d}s".format(int(eh), int(em), int(es)))
            print("lr: ", getLR(opt))
            print("TOP1: L-train loss:{:.5f} / L-acc:{:.5f}".format(train_loss / (ix + 1),
                                                                    100 * top1_correct / total))
            print("TOP5: L-train loss:{:.5f} / L-acc:{:.5f}".format(train_loss / (ix + 1),
                                                                    100 * top5_correct / total))
            print("saving model when training")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), 'epoch': ep}, 'train_last.pt')
            record["train_loss"].append("{:.5f}".format(train_loss / (ix + 1)))
            record["train_top1_acc"].append("{:.5f}".format(100 * top1_correct / total))
            record["train_top5_acc"].append("{:.5f}".format(100 * top5_correct / total))


def test(ep):
    model.eval()
    total = 0
    top1_correct = 0
    top5_correct = 0
    test_loss = 0
    for ix, sample in enumerate(testloader):
        label = sample["label"].cuda()
        data_fast = sample["fast"]["data"][:, :c, :, :, :]
        data_slow = sample["slow"]["data"][:, :c, :, :, :]
        data_fast = data_fast.cuda()
        data_slow = data_slow.cuda()
        output = model(x_fast=data_fast, x_slow=data_slow)

        loss = criterion(output, label)
        test_loss += loss.item()
        _, indices = torch.sort(output.detach(), descending=True)
        top1_correct += indices[:, 0].eq(label).sum().item()
        top5_correct += indices[:, :5].eq(label.view(-1, 1).repeat(1, 5)).sum().item()
        total += label.size(0)
        if (ix + 1) % 10 == 0:
            print("processing testing data {}/{}".format(ix + 1, len(testloader)))

    print("epoch: {}".format(ep))
    print("TOP1: L-test loss:{:.5f} / L-acc:{:.5f}".format(test_loss / (ix + 1),
                                                           100 * top1_correct / total))
    print("TOP5: L-test loss:{:.5f} / L-acc:{:.5f}".format(test_loss / (ix + 1),
                                                           100 * top5_correct / total))
    print("saving model when testing")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), 'epoch': ep}, 'test_last.pt')
    record["test_loss"].append("{:.5f}".format(test_loss / (ix + 1)))
    record["test_top1_acc"].append("{:.5f}".format(100 * top1_correct / total))
    record["test_top5_acc"].append("{:.5f}".format(100 * top5_correct / total))


def getLR(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    # arg
    epoch = 80  # epoch
    bs_train = 16  # batch size for train
    bs_test = 16  # batch size for test
    lr = 0.01  # learning rate
    c = 3  # channel
    alpha = 8  # ratio of sr to fr
    fn = 120  # num of frame for fast stream
    fr = 1  # frame rate for fast stream
    sr = fr * alpha  # frame rate for slow stream
    v = 25  # vortex
    m = 2  # max person in frame
    num_class = 60  # num of class
    first_out_channel = 32  # base feature after the first layer
    data2json = False  # false = (train/test) or true = (prepare json)
    record = {"train_loss": [],
              "test_loss": [],
              "train_top1_acc": [],
              "train_top5_acc": [],
              "test_top1_acc": [],
              "test_top5_acc": []}

    # data
    trainset = Skeleton_Datasets(root_path='/data/chou/Desktop/nturgbd/nturgbd_skeleton',
                                 istrain=True,
                                 mode='cv',
                                 max_human_in_clips=m,
                                 fast_frames_num=fn,
                                 slow_rate=sr,
                                 fast_rate=fr,
                                 preprocess=data2json
                                 )
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=bs_train,
                                              shuffle=True,
                                              num_workers=12)
    testset = Skeleton_Datasets(root_path='/data/chou/Desktop/nturgbd/nturgbd_skeleton',
                                istrain=False,
                                mode='cv',
                                max_human_in_clips=m,
                                fast_frames_num=fn,
                                slow_rate=sr,
                                fast_rate=fr,
                                preprocess=data2json
                                )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=bs_test,
                                             shuffle=False,
                                             num_workers=12)

    print("training video ~ ", len(trainloader) * bs_train)
    print("testing video ~ ", len(testloader) * bs_test)
    # model
    Graphcfg = {"layout": 'ntu-rgb+d', "strategy": 'spatial'}
    model = SlowFastGCN(
        in_channels=c,
        num_class=num_class,
        graph_cfg=Graphcfg,
        edge_importance_weighting=True,
        alpha=alpha,
        k=2,
        baseFeature=first_out_channel
    )
    model = model.cuda()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # train & test
    for i in range(1, epoch + 1):
        print("[TRAINING]")
        train(ep=i)

        if (i % 10) == 0:  # testing every 10 epochs
            print("[TESTING]")
            test(ep=i)
        scheduler.step()  # decay lr every 10 epochs

        with open('record.txt', 'w') as f:
            json.dump(record, f)
