import torch
import torch.nn as nn
import torch.nn.functional as F
from gconv_origin import ConvTemporalGraphical
from graph import Graph
import collections


class ZeroLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0


class IdenLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EdgeImportanceLayer(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.EdgeImportance = nn.Parameter(torch.ones(A.size()))

    def forward(self, A):
        return self.EdgeImportance * A


class DataNormLayer(nn.Module):
    def __init__(self, in_channels, A, data_bn=True):
        super().__init__()
        if data_bn:
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = IdenLayer()

    def forward(self, x):
        N, C, T, V, M = x.size()  # (batch, channel, time, vortex, man)
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # continuous in memory
        x = x.view(N * M, V * C, T)  # aggregate
        x = self.data_bn(x)  # batchnorm
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # like (N, C, H, W) in image
        x = x.view(N * M, C, T, V)
        return x


class SlowFastGCN(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        data_bn: normalize input or not
        alpha: ratio of fast and slow
        beta: ratio of fast sample rate to and slow sample rate
        k: adjust the ratio
        baseFeature: the first output channel
        layers: num of layers in each block(total 3 blocks)


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
                 beta=8,
                 k=1,
                 baseFeature=8,
                 layers=[4, 3, 3]):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)  # just produce a adjMatrix
        A = torch.tensor(self.graph.A,  # graph.A
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # kernel size
        spatial_kernel_size = A.size(0)  # numbe of adjMatrix e.g. 1, 3
        temporal_kernel_size = 9  # conv size for fast time
        temporal_kernel_size_slow = 3  # conv size for slow time
        kernel_size_fast = (temporal_kernel_size, spatial_kernel_size)
        kernel_size_slow = (temporal_kernel_size_slow, spatial_kernel_size)

        # layer for preprocessing data and normalize it
        self.data_norm = DataNormLayer(in_channels, A, data_bn=True)

        # build fast and lateral block
        fast_d, lateral_d = collections.OrderedDict(), collections.OrderedDict()
        in_c_fast = 3
        out_c_fast = basefeature
        is_residual = False
        for layer_id, layer_size in enumerate(layers):
            if layer_id != 0:
                out_c_fast = in_c_fast * 2

            for i in range(layer_size):
                tmp_layer = st_gcn_block(in_c_fast, out_c_fast, kernel_size_fast, A, 1, residual=is_residual)
                in_c_fast = out_c_fast
                fast_d['stgcn_{}_{}'.format(layer_id, i)] = tmp_layer
                is_residual = True

            if layer_id != len(layers) - 1:
                lateral_d['lateral_{}'.format(layer_id)] = nn.Conv2d(out_c_fast,
                                                                     out_c_fast * k * beta,
                                                                     kernel_size=(5, 1),
                                                                     stride=(beta, 1),
                                                                     padding=(2, 0),
                                                                     bias=False)

        self.fast_module = nn.Sequential(fast_d)
        self.name_fast_module = [(name, layer) for name, layer in self.fast_module.named_modules() if (name != '') and (len(name.split(".")) == 1)]
        self.lateral_module = nn.Sequential(lateral_d)
        self.name_lateral_module = [(name, layer) for name, layer in self.lateral_module.named_modules() if (name != '') and (len(name.split(".")) == 1)]

        # bild slow block
        slow_d = collections.OrderedDict()
        in_c_slow = 3
        out_c_fast = alpha * basefeature
        is_residual = False
        for layer_id, layer_size in enumerate(layers):
            if layer_id != 0:
                out_c_fast = in_c_slow * 2
                # not sure
                in_c_slow = in_c_slow + (in_c_slow // alpha) * beta * k

            for i in range(layer_size):
                tmp_layer = st_gcn_block(in_c_slow, out_c_fast, kernel_size_slow, A, 1, residual=is_residual)
                in_c_slow = out_c_fast
                slow_d['stgcn_{}_{}'.format(layer_id, i)] = tmp_layer
                is_residual = True

        self.slow_module = nn.Sequential(slow_d)
        self.name_slow_module = [(name, layer) for name, layer in self.slow_module.named_modules() if (name != '') and (len(name.split(".")) == 1)]
        self.total_layers = len(self.name_slow_module)
        self.fcn = nn.Conv2d(out_c_fast + in_c_fast, num_class, kernel_size=1)

    def forward(self, x_fast, x_slow):
        # preporcess input for fast stream
        N, C, T, V, M = x_fast.size()
        x_fast = self.data_norm(x_fast)  # NM, C, T, V
        x_slow = self.data_norm(x_slow)  # NM, C, T, V

        prvs_block = "0"
        for i in range(self.total_layers):
            slow_name, slow_layer = self.name_slow_module[i]
            fast_name, fast_layer = self.name_fast_module[i]
            now_block = fast_name.split('_')[1]
            if now_block != prvs_block:
                _, lateral_layer = self.name_lateral_module[int(prvs_block)]
                x_slow = torch.cat([lateral_layer(x_fast), x_slow], dim=1)
                prvs_block = now_block

            x_fast = fast_layer({"x": x_fast, "A": self.A})
            x_slow = slow_layer({"x": x_slow, "A": self.A})
            print(x_fast.size(), x_slow.size())

        # global pooling for fast and slow stream
        x_fast = F.avg_pool2d(x_fast, x_fast.size()[2:])
        x_slow = F.avg_pool2d(x_slow, x_slow.size()[2:])

        # # merge two streams
        x = torch.cat([x_fast, x_slow], dim=1)
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        print(x.size())

        # # class prediction
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
                 A,
                 stride=1,
                 dropout=0,
                 residual=True,):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1  # ensure time is odd
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.EdgeImportance = EdgeImportanceLayer(A)  # edge layer for adjMatrix A

        # view tv as hw (2D conv), kernel_size[1] means number of adjMatrix = GCN
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        # just adjust the dimension of time = TCN
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
            self.residual = ZeroLayer()

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = IdenLayer()

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        x = X['x']
        A = X['A']
        # input x, and A
        weight_A = self.EdgeImportance(A)  # weighting A with EdgeImportance layer
        res = self.residual(x)  # shortcut
        x = self.gcn(x, weight_A)  # gcn
        x = self.tcn(x) + res  # tcn

        return self.relu(x)


if __name__ == "__main__":
    num_class = 60  # classes
    bs = 1  # batch size
    m = 2  # people
    c = 3  # channel
    alpha = 8  # channel ratio of fast/slow
    beta = 8  # frame ratio of fast/slow
    k = 1  # adjusting fast channel for concat (c -> k*beta*c)
    fast_frame = 120  # frame number
    v = 25  # points of skeleton
    basefeature = 8  # baseFeature
    layers = [4, 3, 3]  # layer number in each block (suppose that we have 3 block in two stream)
    data_fast = torch.zeros((bs, c, fast_frame, v, m), dtype=torch.float32)
    data_slow = torch.zeros((bs, c, fast_frame // beta, v, m), dtype=torch.float32)
    Graphcfg = {"layout": 'ntu-rgb+d', "strategy": 'spatial'}
    model = SlowFastGCN(
        in_channels=c,
        num_class=num_class,
        graph_cfg=Graphcfg,
        edge_importance_weighting=True,
        alpha=alpha,
        beta=beta,
        k=k,
        baseFeature=basefeature,
        layers=layers
    )
    print("fast size: ", data_fast.size(), "slow size: ", data_slow.size())
    output = model(x_fast=data_fast, x_slow=data_slow)
    print(output.size())
