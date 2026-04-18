import torch
from torch import nn
import pdb


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, kernel_size=3, padding=1, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, kernel_size=kernel_size, padding=padding))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0,normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    

class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode="trilinear",align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        upsampling = UpsamplingDeconvBlock ## using transposed convolution

        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = upsampling(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = upsampling(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        # print(f'x5_up: {x5_up.shape}, x4: {x4.shape}')
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, x8_up
 
class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        dim_in = 16
        feat_dim = 32
        self.pool = nn.MaxPool3d(3, stride=2)
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim))

    def forward(self, input):
        features = self.encoder(input)
        out_seg, x8_up = self.decoder(features)
        # features = self.pool(features[4])
        return out_seg, features # 4, 16, 112, 112, 80

import torch.nn.functional as F
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果输入输出通道不同，则使用1x1卷积进行匹配
        self.match_channels = nn.Conv1d(in_channels, out_channels, 1, stride=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果通道不匹配，调整residual
        if self.match_channels:
            residual = self.match_channels(residual)

        out += residual  # 跳跃连接
        out = self.relu(out)
        return out


class FeatureProcessor1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureProcessor1, self).__init__()
        # 两层ResNet块
        self.resnet_block1 = ResNetBlock(in_channels, out_channels)
        self.resnet_block2 = ResNetBlock(out_channels, out_channels)
        # 用线性层来调整最后的形状
        self.fc = nn.Linear(4096, 7 * 7 * 5)
        # self.fc = nn.Linear(4096, target_shape[-1] * target_shape[-2] * target_shape[-3])  # 将最后的4096展平成(7*7*5)

    def forward(self, diffs):
        x = diffs
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), 7, 7, 5)
        # 调整形状，将4096变成目标形状(7*7*5)
        return x


class FeatureProcessor(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super(FeatureProcessor, self).__init__()
        self.resnet_block1 = ResNetBlock(in_channels, out_channels)
        self.resnet_block2 = ResNetBlock(out_channels, out_channels) # Compute total elements
        self.resnet_block3 = ResNetBlock(in_channels, out_channels)
        # self.fc = nn.Linear(4096, 1024)
        # self.fc2 = nn.Linear(1024, 7 * 7 * 5)
        self.fc = nn.Linear(4096, 7 * 7 * 5)
        self.relu = nn.ReLU()


    def forward(self, diffs):
        x = self.resnet_block1(diffs)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.fc(x)
        x = self.relu(x)
        x = x.view(x.size(0), x.size(1), 7, 7, 5)

        return x


#############代码1
class VNet_CF(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet_CF, self).__init__()

        # self.labelbs = labeled
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)



        output_shape = (7, 7, 5)
        self.processor1 = FeatureProcessor(in_channels=256, out_channels=256, output_shape=output_shape)
        self.processor2 = FeatureProcessor(in_channels=256, out_channels=256, output_shape=output_shape)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))  # 控制特征交互的权重

    def forward(self, image_input, text_features=None):
        batch_size = image_input.size(0)
        half_batch_size = batch_size // 2

        # Step 1: 正常输出
        encoder_features = self.encoder(image_input)  # 得到编码器的多层特征
        last_features = encoder_features[-1]
        # print(encoder_features[-1].shape)        ## torch.Size([4, 256, 7, 7, 5])       #### torch.Size([4, 256, 17, 17, 12])
        output, _ = self.decoder(encoder_features)

        if text_features is not None:
            # Step 2: 反事实输出
            ## Step 2.1: 计算差异性
            diffs = []
            for i in range(half_batch_size):
                diff = torch.abs(text_features[i] - text_features[half_batch_size + i])  ## torch.sie([256, 4096])
                diffs.append(diff)
            diffs = torch.stack(diffs, dim=0)  ## ## torch.sie([2, 256, 4096])

            ## Step 2.2: 计算差异方向
            T_l2u = self.processor1(diffs)
            T_u2l = self.processor2(diffs)
            T = torch.cat((T_u2l, T_l2u))

            ## Step 2.3: 反事实特征，标记数据的未标记数据特征互相生成
            last_features_1 = torch.cat([last_features[half_batch_size:], last_features[:half_batch_size]]) + self.beta * T
            encoder_features[-1] = last_features_1
            ## Step 2.4: 将互相生成的特征交换为标记数据和无标记数据
            counterfactual_output, _ = self.decoder1(encoder_features)

            ## 反事实特征的反事实特征
            last_features_2 = last_features + self.alpha * T
            encoder_features[-1] = last_features_2
            ## Step 2.4: 将互相生成的特征交换为标记数据和无标记数据
            counterfactual_output_counterfactual, _ = self.decoder2(encoder_features)
            return output, counterfactual_output, counterfactual_output_counterfactual
        else:
            return output

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = VNet(n_channels=1, n_classes=1, normalization='batchnorm', has_dropout=False)
    input = torch.randn(1, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)

    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #   macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
    #                                            print_per_layer_stat=True, verbose=True)
    #   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #   print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #import pdb; pdb.set_trace()
