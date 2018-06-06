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

        self.output = nn.Linear(filters, 1)

    def forward(self, x):
        x = x.cuda().double()
        x = self.initialconv(x)
        x = self.initialbn(x)

        for i in range(self.layers):
            x = F.relu(getattr(self,'conv_{}'.format(i))(x))
            x = getattr(self,'bn_{}'.format(i))(x)
            x = F.max_pool1d(x, kernel_size=3, stride=3)

        x = F.max_pool1d(x, kernel_size=x.size()[2:])
        x = x.view(-1, self.filters)
        x = F.sigmoid(self.output(x))

        return x


class SimpleNet(nn.Module):
    def __init__(self, filters=128):
        super(SimpleNet, self).__init__()
        self.filters = filters

        self.conv0 = nn.Conv1d(1, filters, 3, padding=1)
        self.conv0bn = nn.BatchNorm1d(filters)
        self.conv1 = nn.Conv1d(filters, filters, 3, padding=1)
        self.conv1bn = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, 3, padding=1)
        self.conv2bn = nn.BatchNorm1d(filters)
        self.conv3 = nn.Conv1d(filters, 2 * filters, 3, padding=1)
        self.conv3bn = nn.BatchNorm1d(2 * filters)
        self.conv4 = nn.Conv1d(2 * filters, 2 * filters, 3, padding=1)
        self.conv4bn = nn.BatchNorm1d(2 * filters)
        self.conv5 = nn.Conv1d(2 * filters, 4 * filters, 3, padding=1)
        self.conv5bn = nn.BatchNorm1d(4 * filters)
        self.conv6 = nn.Conv1d(4 * filters, 4 * filters, 3, padding=1)
        self.conv6bn = nn.BatchNorm1d(4 * filters)
        self.conv7 = nn.Conv1d(4 * filters, 4 * filters, 3, padding=1)
        self.conv7bn = nn.BatchNorm1d(4 * filters)

        self.fc = nn.Linear(512, 512)
        self.conv0bn = nn.BatchNorm1d(filters)

        self.output = nn.Linear(512, 1)

    def forward(self, x):
        x = x.cuda().double()

        x = F.relu(self.conv0(x))
        x = self.conv0bn(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = self.conv1bn(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = self.conv2bn(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = self.conv3bn(x)
        x = F.relu(self.conv4(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = self.conv4bn(x)
        x = F.relu(self.conv5(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = self.conv5bn(x)
        x = F.relu(self.conv6(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = self.conv6bn(x)
        x = F.relu(self.conv7(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = self.conv7bn(x)
        x = F.max_pool1d(x, kernel_size=x.size()[2])
        x = x.view(-1, self.filters * 4)
        x = F.relu(self.fc(x))
        x = F.sigmoid(self.output(x))

        return x