import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(ResNet, self).__init__()

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError("Unsupported output stride:", output_stride)

        # Define layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3, stride=strides[0])
        self.layer2 = self._make_layer(256, 128, 4, stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(512, 256, 6, stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(1024, 512, 3, stride=strides[3], dilation=dilations[3])


    def _make_layer(self, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = []
        layers.append(ResidualBlock(inplanes, planes, stride, dilation, downsample))
        for i in range(1, blocks):
            layers.append(ResidualBlock(planes * 4, planes, 1, dilation))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features






class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, *args, **kwargs):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        # first conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # third conv block
        out = self.conv3(out)
        out = self.bn3(out)

        # downsample identity to match output size
        if self.downsample is not None:
            identity = self.downsample(x)

        # add identity to output and apply ReLU
        out += identity
        out = self.relu(out)

        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(in_channels + out_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels, n_classes, 3, padding=1)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.norm1(low_level_features)
        low_level_features = self.relu1(low_level_features)

        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)


        x = self.upconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)

        x = F.interpolate(x, x.shape[-1]*2, mode='bilinear', align_corners=True)
        return torch.sigmoid(x)



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.rates = rates

        # 1x1 convolution
        self.conv1x1_1 = nn.Conv2d(out_channels*5, out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # ASPP convolutions
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.conv3x3_4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.bn6 = nn.BatchNorm2d(out_channels)

        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply ASPP convolutions with different dilation rates
        x1 = self.conv3x3_1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv3x3_2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv3x3_3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        x4 = self.conv3x3_4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)


        # Apply global average pooling to input feature map
        x6 = F.adaptive_avg_pool2d(x, 1)
        x6 = self.conv1x1_2(x6)
        x6 = self.bn5(x6)
        x6 = self.relu(x6)
        x6 = F.interpolate(x6, size=x.size()[2:], mode='bilinear', align_corners=True)

        # Concatenate the results
        out = self.relu(self.bn6(self.conv1x1_1(torch.cat([x1, x2, x3, x4, x6], dim=1))))
        return out




class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes, output_stride=8):
        super(DeepLabV3Plus, self).__init__()

        # ResNet50 backbone
        self.backbone = ResNet(3,output_stride=output_stride)

        # ASPP module
        self.aspp = ASPP(in_channels=2048, out_channels=256, rates=[6, 12, 18])

        # Decoder module
        self.decoder = Decoder(in_channels=256, out_channels=48, n_classes=n_classes)

    def forward(self, x):
        # Backbone
        x, low_level_features = self.backbone(x)



        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x, low_level_features)


        return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

model = DeepLabV3Plus(1)
model.apply(init_weights)





if __name__ == "__main__":
    input = torch.rand(2, 3, 500, 500)
    print(model)
    output = model(input)
    print(output.shape)
