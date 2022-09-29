from mmcv.ops import DeformConv2dPack as DCN
import torch
import torch.nn as nn

class DCNNet(nn.Module):
    def __init__(self, args):
        super(DCNNet,self).__init__()

        module_list =[]
        in_channels = args['in_channels']
        out_channels = args['out_channels']
        stride = args['stride']
        kernel_size = args['kernel_size']
        padding = args['padding']

        for i in range(args['n_blocks']):
            module_list.append(DCN(in_channels[i],out_channels[i],kernel_size[i],stride=stride[i],padding=padding[i]))
        self.model = nn.Sequential(*module_list)

    def forward(self, x):
        return self.model(x)