import torch
import torch.nn as nn
import torch.nn.functional as F
from gconv_origin import ConvTemporalGraphical
from graph import Graph


def zero(x):
    return 0


def iden(x):
    return x


class EdgeImportanceLayer(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.EdgeImportance = nn.Parameter(torch.ones(A.size()))

    def forward(self, x):
        return self.EdgeImportance * x


class SlowFastGCN(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=False,
                 data_bn=True,
                 alpha=8,
                 k=2,
                 baseFeature=8,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)  # just produce a adjMatrix
        A = torch.tensor(self.graph.A,  # graph.A
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)  # 1 or 3 (depend on adjMatrix)
        temporal_kernel_size = 13  # conv size for fast time
        temporal_kernel_size_slow = 3  # conv size for slow time
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kernel_size_slow = (temporal_kernel_size_slow, spatial_kernel_size)

        # normalize data or not
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        # a gcn stream for fast data
        self.basefeature = baseFeature
        self.alpha = alpha
        self.k = k
        self.st_gcn_networks_fast = nn.ModuleList((
            st_gcn_block(in_channels, self.basefeature, kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(self.basefeature, self.basefeature, kernel_size, 1, **kwargs),
            st_gcn_block(self.basefeature, self.basefeature, kernel_size, 1, **kwargs),
            st_gcn_block(self.basefeature, self.basefeature, kernel_size, 1, **kwargs),
            # change time here or not
            st_gcn_block(self.basefeature, self.basefeature * 2, kernel_size, 1, **kwargs),
            st_gcn_block(self.basefeature * 2, self.basefeature * 2, kernel_size, 1, **kwargs),
            st_gcn_block(self.basefeature * 2, self.basefeature * 2, kernel_size, 1, **kwargs),
            st_gcn_block(self.basefeature * 2, self.basefeature * 4, kernel_size, 1, **kwargs),
            st_gcn_block(self.basefeature * 4, self.basefeature * 4, kernel_size, 1, **kwargs),
            st_gcn_block(self.basefeature * 4, self.basefeature * 4, kernel_size, 1, **kwargs),
        ))

        # a gcn stream for slow data
        self.st_gcn_networks_slow = nn.ModuleList((
            st_gcn_block(in_channels, self.basefeature * self.alpha, kernel_size_slow, 1, residual=False, **kwargs0),
            st_gcn_block(self.basefeature * self.alpha, self.basefeature * self.alpha, kernel_size_slow, 1, **kwargs),
            st_gcn_block(self.basefeature * self.alpha, self.basefeature * self.alpha, kernel_size_slow, 1, **kwargs),
            st_gcn_block(self.basefeature * self.alpha, self.basefeature * self.alpha, kernel_size_slow, 1, **kwargs),
            # change time here or not
            st_gcn_block(self.basefeature * self.alpha + self.basefeature * 2, self.basefeature * self.alpha * 2, kernel_size_slow, 1, **kwargs),
            st_gcn_block(self.basefeature * self.alpha * 2, self.basefeature * self.alpha * 2, kernel_size_slow, 1, **kwargs),
            st_gcn_block(self.basefeature * self.alpha * 2, self.basefeature * self.alpha * 2, kernel_size_slow, 1, **kwargs),
            st_gcn_block(self.basefeature * self.alpha * 2 + self.basefeature * 4, self.basefeature * self.alpha * 4, kernel_size_slow, 1, **kwargs),
            st_gcn_block(self.basefeature * self.alpha * 4, self.basefeature * self.alpha * 4, kernel_size_slow, 1, **kwargs),
            st_gcn_block(self.basefeature * self.alpha * 4, self.basefeature * self.alpha * 4, kernel_size_slow, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            print("adjacency weight mask added!!!")
            self.edge_importance_fast = nn.ModuleList([EdgeImportanceLayer(A) for i in self.st_gcn_networks_fast])
            # self.edge_importance_fast = nn.ParameterList([
            #     nn.Parameter(torch.ones(self.A.size()))
            #     for i in self.st_gcn_networks_fast
            # ])
            self.edge_importance_slow = nn.ModuleList([EdgeImportanceLayer(A) for i in self.st_gcn_networks_slow])
            # self.edge_importance_slow = nn.ParameterList([
            #     nn.Parameter(torch.ones(self.A.size()))
            #     for i in self.st_gcn_networks_slow
            # ])
        else:
            self.edge_importance_fast = [1] * len(self.st_gcn_networks_fast)
            self.edge_importance_slow = [1] * len(self.st_gcn_networks_slow)

        # fcn for prediction
        self.fcn = nn.Conv2d(self.basefeature * self.alpha * 4 + self.basefeature * 4, num_class, kernel_size=1)
        # self.fcn = nn.Conv2d(self.basefeature*4, num_class, kernel_size=1)

        # for concat
        self.lateralA = nn.Conv2d(self.basefeature,
                                  self.basefeature * 2,
                                  kernel_size=(5, 1),
                                  stride=(self.alpha, 1),
                                  padding=(2, 0),
                                  bias=False
                                  )
        self.lateralB = nn.Conv2d(self.basefeature * 2,
                                  self.basefeature * 2 * 2,
                                  kernel_size=(5, 1),
                                  stride=(self.alpha, 1),
                                  padding=(2, 0),
                                  bias=False
                                  )

    def forward(self, x_fast, x_slow):
        N, C, T, V, M = x_fast.size()  # (batch, channel, time, vortex, man)
        x_fast = x_fast.permute(0, 4, 3, 1, 2).contiguous()  # continuous in memory
        x_fast = x_fast.view(N * M, V * C, T)  # aggregate
        x_fast = self.data_bn(x_fast)  # batchnorm
        x_fast = x_fast.view(N, M, V, C, T)  # span
        x_fast = x_fast.permute(0, 1, 3, 4, 2).contiguous()  # just like (N, C, H, W) in image
        x_fast = x_fast.view(N * M, C, T, V)

        N, C, T, V, M = x_slow.size()
        x_slow = x_slow.permute(0, 4, 3, 1, 2).contiguous()
        x_slow = x_slow.view(N * M, V * C, T)
        x_slow = self.data_bn(x_slow)
        x_slow = x_slow.view(N, M, V, C, T)
        x_slow = x_slow.permute(0, 1, 3, 4, 2).contiguous()
        x_slow = x_slow.view(N * M, C, T, V)

        # fast stream
        lateral = []
        for ix, (gcn, importance_layer) in enumerate(zip(self.st_gcn_networks_fast, self.edge_importance_fast)):
            importance = importance_layer(self.A)
            x_fast = gcn(x_fast, importance)
            # x_fast, _ = gcn(x_fast, self.A * importance)
            print('fast', ix, x_fast.size())
            if ix == 3:
                lateral.append(self.lateralA(x_fast))
            if ix == 6:
                lateral.append(self.lateralB(x_fast))

        # slow stream
        for ix, (gcn, importance_layer) in enumerate(zip(self.st_gcn_networks_slow, self.edge_importance_slow)):
            importance = importance_layer(self.A)
            x_slow = gcn(x_slow, importance)
            # x_slow, _ = gcn(x_slow, self.A * importance)
            print('slow', ix, x_slow.size())
            if ix == 3:
                x_slow = torch.cat([x_slow, lateral[0]], dim=1)
            if ix == 6:
                x_slow = torch.cat([x_slow, lateral[1]], dim=1)

        # global pooling
        x_fast = F.avg_pool2d(x_fast, x_fast.size()[2:])
        x_slow = F.avg_pool2d(x_slow, x_slow.size()[2:])

        # merge two streams
        x = torch.cat([x_fast, x_slow], dim=1)
        # print(x.size())
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction class
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1  # ensure time is odd
        padding = ((kernel_size[0] - 1) // 2, 0)

        # view tv as hw (2D conv), kernel_size[1] means number of adjMatrix
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        # just adjust the dimension of time
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x)


if __name__ == "__main__":
    num_class = 60
    bs = 1
    m = 2
    c = 3
    alpha = 8
    fast_frame = 120
    v = 25
    basefeature = 16
    data_fast = torch.zeros((bs, c, fast_frame, v, m), dtype=torch.float32)
    data_slow = torch.zeros((bs, c, fast_frame // alpha, v, m), dtype=torch.float32)
    Graphcfg = {"layout": 'ntu-rgb+d', "strategy": 'spatial'}
    model = SlowFastGCN(
        in_channels=c,
        num_class=num_class,
        graph_cfg=Graphcfg,
        edge_importance_weighting=True,
        alpha=alpha,
        baseFeature=basefeature
    )
    print("fast before model: ", data_fast.size())
    print("slow before model: ", data_slow.size())
    output = model(x_fast=data_fast, x_slow=data_slow)
    print(output.size())
