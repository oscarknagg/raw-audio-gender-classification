import torch.nn as nn
import torch.nn.functional as F


class DilatedNet(nn.Module):
    def __init__(self, filters, dilation_depth, dilation_stacks):
        super(DilatedNet, self).__init__()
        self.filters = filters
        self.dilation_depth = dilation_depth
        self.dilation_stacks = dilation_stacks
        self.receptive_field = dilation_stacks*(3**dilation_depth)

        self.initialconv = nn.Conv1d(1, filters, 3, dilation=1, padding=1)

        for s in range(dilation_stacks):
            for i in range(dilation_depth):
                setattr(
                    self,
                    'dilated_conv_{}_relu_s{}'.format(3 ** i, s),
                    nn.Conv1d(filters, filters, 3, dilation=3 ** i, padding=3 ** i)
                )

        self.finalconv = nn.Conv1d(filters, filters, 3, dilation=1, padding=1)
        self.output = nn.Linear(filters, 1)

    def forward(self, x):
        x = x.cuda().double()
        x = self.initialconv(x)

        skip_connections = []
        for s in range(self.dilation_stacks):
            for i in range(self.dilation_depth):
                original_x = x
                x = F.relu(getattr(self, 'dilated_conv_{}_relu_s{}'.format(3 ** i, s))(x))
                skip_connections.append(x)
                x = x + original_x

        x = F.relu(self.finalconv(x))

        x = F.max_pool1d(x, kernel_size=x.size()[2:])
        x = x.view(-1, self.filters)
        x = F.sigmoid(self.output(x))
        return x


class ConvNet(nn.Module):
    def __init__(self, filters, layers):
        super(ConvNet, self).__init__()
        self.filters = filters
        self.layers = layers
        self.receptive_field = 3**layers

        self.initialconv = nn.Conv1d(1, filters, 3, dilation=1, padding=1)
        self.initialbn = nn.BatchNorm1d(filters)

        for i in range(layers):
            setattr(
                self,
                'conv_{}'.format(i),
                nn.Conv1d(filters, filters, 3, dilation=1, padding=1)
            )
            setattr(
                self,
                'bn_{}'.format(i),
                nn.BatchNorm1d(filters)
            )

        self.finalconv = nn.Conv1d(filters, filters, 3, dilation=1, padding=1)

        self.output = nn.Linear(filters, 1)

    def forward(self, x):
        x = x.cuda().double()
        x = self.initialconv(x)
        x = self.initialbn(x)

        for i in range(self.layers):
            x = F.relu(getattr(self,'conv_{}'.format(i))(x))
            x = getattr(self,'bn_{}'.format(i))(x)
            x = F.max_pool1d(x, kernel_size=3, stride=3)

        x = F.relu(self.finalconv(x))

        x = F.max_pool1d(x, kernel_size=x.size()[2:])
        x = x.view(-1, self.filters)
        x = F.sigmoid(self.output(x))

        return x
