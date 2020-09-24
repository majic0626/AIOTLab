# The based unit of graph convolutional networks.
# This is the original implementation for ST-GCN papers.

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution

        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0

        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
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
                 num_adjMatrix,  # 3 or 1
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.num_adjMatrix = num_adjMatrix
        self.conv = nn.Conv2d(in_channels,
                              out_channels * num_adjMatrix,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        """
        Basically, This is Conv2D works on (N, C, T=H, V=W)
        kernel_size is number of AdjMatrix, e.g. 3
        output channel after Conv2D = out_channels * kernel_size

        """
        x = self.conv(x)
        n, kc, t, v = x.size()  # look c -> 3*c
        # x: Batch , A.size(0) , Channels , Time , Points
        # A: A.size(0) , Points , Points
        # goal: Batch , Channels , Time , Points

        """ 20200903 """
        x = x.view(n, self.num_adjMatrix, kc // self.num_adjMatrix, t, v)  # split Adj size and Channels
        x = x.permute(0, 3, 1, 4, 2).contiguous()  # N T K V C
        A_repeat = A.repeat(n, t, 1, 1, 1).clone()  # N T K V V

        x = x.view(n * t * self.num_adjMatrix, v, kc // self.num_adjMatrix)  # (NTK) V C
        A_repeat = A_repeat.view(n * t * self.num_adjMatrix, v, v)  # (NTK) V V
        x = torch.bmm(A_repeat, x)  # size: (NTK) V C
        x = x.view(n, t, self.num_adjMatrix, v, -1)
        x = torch.mean(x, dim=2)  # N T V C
        x = x.permute(0, 3, 1, 2).contiguous()

        return x
