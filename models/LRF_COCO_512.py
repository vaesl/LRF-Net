import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class LDS(nn.Module):
    def __init__(self,):
        super(LDS, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

    def forward(self, x):
        x_pool1 = self.pool1(x)
        x_pool2 = self.pool2(x_pool1)
        x_pool3 = self.pool3(x_pool2)
        return x_pool3


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class LSN_init(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(LSN_init, self).__init__()
        self.out_channels = out_planes
        inter_planes = out_planes // 4
        self.part_a = nn.Sequential(
                ConvBlock(in_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1),
                ConvBlock(inter_planes, inter_planes, kernel_size=1, stride=1),
                ConvBlock(inter_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1)
                )
        self.part_b = ConvBlock(inter_planes, out_planes, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out1 = self.part_a(x)
        out2 = self.part_b(out1)
        return out1, out2


class LSN_later(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(LSN_later, self).__init__()
        self.out_channels = out_planes
        inter_planes = out_planes // 4
        self.part_a = ConvBlock(in_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1)
        self.part_b = ConvBlock(inter_planes, out_planes, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out1 = self.part_a(x)
        out2 = self.part_b(out1)
        return out1, out2


class IBN(nn.Module):

    def __init__(self, out_planes, bn=True):
        super(IBN, self).__init__()
        self.out_channels = out_planes
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        return x


class One_Three_Conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(One_Three_Conv, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.single_branch = nn.Sequential(
                ConvBlock(in_planes, inter_planes, kernel_size=1, stride=1),
                ConvBlock(inter_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=1, relu=False)
                )

    def forward(self, x):
        out = self.single_branch(x)
        return out


class Relu_Conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Relu_Conv, self).__init__()
        self.out_channels = out_planes
        self.relu = nn.ReLU(inplace=False)
        self.single_branch = nn.Sequential(
            ConvBlock(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=1)
        )

    def forward(self, x):
        x = self.relu(x)
        out = self.single_branch(x)
        return out


class Ds_Conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, padding=(1, 1)):
        super(Ds_Conv, self).__init__()
        self.out_channels = out_planes
        self.single_branch = nn.Sequential(
            ConvBlock(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=padding, relu=False)
        )

    def forward(self, x):
        out = self.single_branch(x)
        return out


class LRFNet(nn.Module):
    """LRFNet for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(LRFNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        # vgg network
        self.base = nn.ModuleList(base)

        self.lds = LDS()

        # convs for merging the lsn and ssd features
        self.Norm1 = Relu_Conv(512, 512, stride=1)
        self.Norm2 = Relu_Conv(1024, 1024, stride=1)
        self.Norm3 = Relu_Conv(512, 512, stride=1)
        self.Norm4 = Relu_Conv(256, 256, stride=1)
        self.Norm5 = Relu_Conv(256, 256, stride=1)

        # convs for generate the lsn features
        self.icn1 = LSN_init(3, 512, stride=1)
        self.icn2 = LSN_later(128, 1024, stride=2)
        self.icn3 = LSN_later(256, 512, stride=2)
        self.icn4 = LSN_later(128, 256, stride=2)

        # convs with s=2 to downsample the features
        self.dsc1 = Ds_Conv(512, 1024, stride=2, padding=(1, 1))
        self.dsc2 = Ds_Conv(1024, 512, stride=2, padding=(1, 1))
        self.dsc3 = Ds_Conv(512, 256, stride=2, padding=(1, 1))
        self.dsc4 = Ds_Conv(256, 256, stride=2, padding=(1, 1))

        # convs to reduce the feature dimensions of current level
        self.agent1 = ConvBlock(512, 256, kernel_size=1, stride=1)
        self.agent2 = ConvBlock(1024, 512, kernel_size=1, stride=1)
        self.agent3 = ConvBlock(512, 256, kernel_size=1, stride=1)
        self.agent4 = ConvBlock(256, 128, kernel_size=1, stride=1)

        # convs to reduce the feature dimensions of other levels
        self.proj1 = ConvBlock(1024, 128, kernel_size=1, stride=1)
        self.proj2 = ConvBlock(512, 128, kernel_size=1, stride=1)
        self.proj3 = ConvBlock(256, 128, kernel_size=1, stride=1)
        self.proj4 = ConvBlock(256, 128, kernel_size=1, stride=1)

        # convs to reduce the feature dimensions of other levels
        self.convert1 = ConvBlock(512, 256, kernel_size=1)
        self.convert2 = ConvBlock(384, 512, kernel_size=1)
        self.convert3 = ConvBlock(256, 256, kernel_size=1)
        self.convert4 = ConvBlock(128, 128, kernel_size=1)

        # convs to merge the features of the current and higher level features
        self.merge1 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.merge2 = ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.merge3 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.merge4 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)

        self.ibn1 = IBN(512, bn=True)
        self.ibn2 = IBN(1024, bn=True)

        self.relu = nn.ReLU(inplace=False)

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax()

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,512,512].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        new_sources = list()

        # apply lds to the initial image
        x_pool = self.lds(x)

        # apply vgg up to conv4_3
        for k in range(22):
            x = self.base[k](x)
        conv4_3_bn = self.ibn1(x)
        x_pool1_skip, x_pool1_icn = self.icn1(x_pool)
        s = self.Norm1(conv4_3_bn * x_pool1_icn)

        # apply vgg up to fc7
        for k in range(22, 34):
            x = self.base[k](x)
        conv7_bn = self.ibn2(x)
        x_pool2_skip, x_pool2_icn = self.icn2(x_pool1_skip)
        p = self.Norm2(self.dsc1(s) + conv7_bn * x_pool2_icn)

        x = self.base[34](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k == 0:
                x_pool3_skip, x_pool3_icn = self.icn3(x_pool2_skip)
                w = self.Norm3(self.dsc2(p) + x * x_pool3_icn)
            elif k == 2:
                x_pool4_skip, x_pool4_icn = self.icn4(x_pool3_skip)
                q = self.Norm4(self.dsc3(w) + x * x_pool4_icn)
            elif k == 4:
                o = self.Norm5(self.dsc4(q) + x)
                sources.append(o)
            elif k == 7 or k == 9:
                sources.append(x)
            else:
                pass

        # project the forward features into lower dimension.
        tmp1 = self.proj1(p)
        tmp2 = self.proj2(w)
        tmp3 = self.proj3(q)
        tmp4 = self.proj4(o)

        # The conv4_3 level
        proj1 = F.upsample(tmp1, scale_factor=2, mode='bilinear')
        proj2 = F.upsample(tmp2, scale_factor=4, mode='bilinear')
        proj3 = F.upsample(tmp3, scale_factor=8, mode='bilinear')
        proj4 = F.upsample(tmp4, scale_factor=16, mode='bilinear')
        proj = torch.cat([proj1, proj2, proj3, proj4], dim=1)

        agent1 = self.agent1(s)
        convert1 = self.convert1(proj)
        pred1 = torch.cat([agent1, convert1], dim=1)
        pred1 = self.merge1(pred1)
        new_sources.append(pred1)

        # The fc_7 level
        proj2 = F.upsample(tmp2, scale_factor=2, mode='bilinear')
        proj3 = F.upsample(tmp3, scale_factor=4, mode='bilinear')
        proj4 = F.upsample(tmp4, scale_factor=8, mode='bilinear')
        proj = torch.cat([proj2, proj3, proj4], dim=1)

        agent2 = self.agent2(p)
        convert2 = self.convert2(proj)
        pred2 = torch.cat([agent2, convert2], dim=1)
        pred2 = self.merge2(pred2)
        new_sources.append(pred2)

        # The conv8 level
        proj3 = F.upsample(tmp3, scale_factor=2, mode='bilinear')
        proj4 = F.upsample(tmp4, scale_factor=4, mode='bilinear')
        proj = torch.cat([proj3, proj4], dim=1)

        agent3 = self.agent3(w)
        convert3 = self.convert3(proj)
        pred3 = torch.cat([agent3, convert3], dim=1)
        pred3 = self.merge3(pred3)
        new_sources.append(pred3)

        # The conv9 level
        proj4 = F.upsample(tmp4, scale_factor=2, mode='bilinear')
        proj = proj4

        agent4 = self.agent4(q)
        convert4 = self.convert4(proj)
        pred4 = torch.cat([agent4, convert4], dim=1)
        pred4 = self.merge4(pred4)
        new_sources.append(pred4)

        for prediction in sources:
            new_sources.append(prediction)

        # apply multibox head to source layers
        for (x, l, c) in zip(new_sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=False), conv7, nn.ReLU(inplace=False)]
    return layers


base = {
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]
}


def add_extras(size, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [One_Three_Conv(in_channels, cfg[k+1], stride=2), nn.ReLU(inplace=False)]
                else:
                    layers += [One_Three_Conv(in_channels, cfg[k+1], stride=2), nn.ReLU(inplace=False)]
        in_channels = v
    layers += [ConvBlock(256, 128, kernel_size=1, stride=1)]
    layers += [ConvBlock(128, 256, kernel_size=4, stride=1, padding=1)]
    return layers


extras = {
    '512': [1024, 'S', 512, 'S', 256, 'S', 256, 'S', 256]}


def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [1, -2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers +=[nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 2
    indicator = 5

    for k, v in enumerate(extra_layers):
        if k < indicator-2 and k % 2 == 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            i += 1

        elif k > indicator-1 and k % 2 != 0:
            loc_layers += [nn.Conv2d(256, cfg[i]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(256, cfg[i]
                                      * num_classes, kernel_size=3, padding=1)]
            i += 1

        else:
            pass

    return vgg, extra_layers, (loc_layers, conf_layers)


mbox = {'512': [6, 6, 6, 6, 6, 4, 4]}
# number of boxes per feature map location


def build_net(phase, size=512, num_classes=81):
    if size != 512:
        print("Error: The input image size is not supported!")
        return

    return LRFNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
                                add_extras(size, extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes)
