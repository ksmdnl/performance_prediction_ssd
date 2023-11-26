import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

upsample_bilinear = lambda x, outsize: F.interpolate(x, outsize, mode='bilinear', align_corners=False)
upsample_nearest = lambda x, outsize: F.interpolate(x, outsize, mode='nearest')
batchnorm_momentum = 0.01 / 2


# During training it makes sense to choose the deterministic option as this way the training is deterministic.
# After training, switching to the non-deterministic option increases the performance.
def upsample(deterministic, x, outsize):
    if deterministic:  # Implement-wise deterministic is activated with model.train() and deactivated with model.eval()
        return upsample_deterministic(x, int(outsize[0] / x.shape[2]))  # This is effectively nearest neighbour
        # return upsample_bilinear(x, outsize) # !! Use bilinear here if you want performance of non-det training !!
    else:
        return upsample_bilinear(x, outsize)


def upsample_deterministic(x, upscale):
    return x[:, :, :, None, :, None].expand(-1, -1, -1, upscale, -1, upscale).reshape(
        x.size(0), x.size(1), x.size(2) * upscale, x.size(3) * upscale)


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))


# This is exactly the same function as _BNReluConv. It seems like using it this way (and not with a default value)
# leads to more reasonable outputs with torchscope. Otherwise, torchscope will multiply, whysoever, the calls to
# nn.ReLU and counts it multiple times...
class _BNActConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False,
                 dilation=1):  # , act_fn=nn.ReLU(inplace=True)):
        super(_BNActConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('act_fn', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3):
        super(_Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNActConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = _BNActConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(self.training, x, skip_size)  # During training deterministic, in evaluation "non-deterministic"
        # x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x


class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn',
                            _BNActConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNActConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_fuse',
                            _BNActConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)
            level = upsample(self.training, level,
                             target_size)  # During training deterministic, in evaluation "non-deterministic"
            # level = upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class _UpsampleBlend(nn.Module):
    def __init__(self, num_maps_in, num_maps_out, use_bn=True):
        super(_UpsampleBlend, self).__init__()
        self.blend_conv = _BNActConv(num_maps_in, num_maps_out, k=3, batch_norm=use_bn)

    def forward(self, x, skip):
        skip_size = skip.size()[2:4]
        x = upsample(self.training, x, skip_size)  # During training deterministic, in evaluation "non-deterministic"
        x = self.blend_conv.forward(x)
        return x


class ResizeConv2D(nn.Module):
    # change for higher performance to bilinear (but not deterministic)
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1,
                 mode='nearest'):
        super().__init__()
        self.scale_factor = stride
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


def resizeconv3x3(in_planes, out_planes, stride=1, mode='nearest'):
    """3x3 convolution with padding and resizing"""
    if stride > 1:
        layers = [nn.Upsample(scale_factor=stride, mode=mode, align_corners=None)]
    else:
        layers = []
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
    layers += [conv]
    return nn.Sequential(*layers)


def _bn_function_factory(conv, norm, act_fn=None):
    def bn_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if act_fn is not None:
            x = act_fn(x)
        return x

    return bn_function


def do_efficient_fwd(block, x, efficient):
    # Checkpoint function seem to make problem for graph creation
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, dilation=1):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        kwargs = {
            "dilation": dilation,
            "padding": dilation,
        }

        self.conv1 = conv3x3(inplanes, planes, stride, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.act1)
        bn_2 = _bn_function_factory(self.conv2, self.bn2)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        act_fn = self.act2(out)

        return act_fn, out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, dilation=1):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if self.use_bn else None
        self.act3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.act1)
        bn_2 = _bn_function_factory(self.conv2, self.bn2, self.act2)
        bn_3 = _bn_function_factory(self.conv3, self.bn3)  # , self.act3) # This is commented out as it is wrong

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)
        out = do_efficient_fwd(bn_3, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        act_fn = self.act3(out)

        return act_fn, out

class BasicBlockDec(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, efficient=True, use_bn=True, lateral=False):
        super().__init__()
        self.use_bn = use_bn

        self.conv2 = resizeconv3x3(inplanes, inplanes)
        self.bn2 = nn.BatchNorm2d(inplanes) if self.use_bn else None
        self.act2 = nn.ReLU(inplace=True)
        self.conv1 = resizeconv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.act1 = nn.ReLU(inplace=True)

        if stride != 1 or inplanes != int(planes * self.expansion):
            layers_ = [ResizeConv2D(inplanes, int(planes * self.expansion), kernel_size=1,
                                    stride=stride, padding=0)]
            if self.use_bn:  # if you want BatchNormalization in your layer
                layers_ += [nn.BatchNorm2d(planes * self.expansion)]
            self.upsample = nn.Sequential(*layers_)  # residual connection
        else:
            self.upsample = None

        self.stride = stride
        self.efficient = efficient
        self.lateral = lateral

    def forward(self, x):
        if self.lateral:
            if x[0].size()[2:4] != x[1].size()[2:4]:
                x[1] = upsample(self.training, x[1], x[0].size()[2:4])
            residual = x[0] + x[1]  # Here x[1] are lateral connections
            x = x[0]
        else:
            residual = x
        bn_2 = _bn_function_factory(self.conv2, self.bn2, self.act2)
        bn_1 = _bn_function_factory(self.conv1, self.bn1)

        out = do_efficient_fwd(bn_2, x, self.efficient)
        out = do_efficient_fwd(bn_1, out, self.efficient)

        if self.upsample is not None:
            residual = self.upsample(x)

        out = out + residual
        act_fn = self.act2(out)
        return act_fn

# Relevant modules for ConvNeXt
class _BNGELUConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, norm=True, bias=False, dilation=1):
        super(_BNGELUConv, self).__init__()
        if norm:
            self.add_module('norm', LayerNorm(num_maps_in, eps=1e-6, data_format="channels_first"))
        self.add_module('gelu', nn.GELU())
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))

class _UpsampleGELU(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, norm=True, k=3):
        super(_UpsampleGELU, self).__init__()
        self.bottleneck = _BNGELUConv(skip_maps_in, num_maps_in, k=1, norm=norm)
        self.blend_conv = _BNGELUConv(num_maps_in, num_maps_out, k=k, norm=norm)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(self.training, x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x

class SpatialPyramidPoolingGELU(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, norm=True):
        super(SpatialPyramidPoolingGELU, self).__init__()
        self.grids = torch.tensor(grids)
        self.square_grid = torch.tensor(square_grid)
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn',
                            _BNGELUConv(num_maps_in, bt_size, k=1, norm=norm))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNGELUConv(num_features, level_size, k=1, norm=norm))
        self.spp.add_module('spp_fuse',
                            _BNGELUConv(final_size, out_size, k=1, norm=norm))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, torch.round(ar * self.grids[i - 1])))
                grid_size = [int(g) for g in grid_size]
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)
            level = upsample(self.training, level,
                             target_size)  # During training deterministic, in evaluation "non-deterministic"
            # level = upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)
        

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)) + self.eps
        x = torch.div(x, norm)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        
        return x