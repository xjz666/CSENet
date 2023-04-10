import torch
import torch.nn as nn
import torchvision.models

bachbone = torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.DEFAULT)




class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x




class Baseline(nn.Module):
    def __init__(self, num_classes=1):
        super(Baseline, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        filters = (128, 256, 512, 1024)
        self.features1 = bachbone.features[:2]
        self.features2 = bachbone.features[2:4]
        self.features3 = bachbone.features[4:6]
        self.features4 = bachbone.features[6:]

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU()
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU()
        self.finalconv3 = nn.Conv2d(32, num_classes, 3)

        self.decoder4.apply(init_weights)
        self.decoder3.apply(init_weights)
        self.decoder2.apply(init_weights)
        self.decoder1.apply(init_weights)
        self.finaldeconv1.apply(init_weights)
        self.finalconv2.apply(init_weights)
        self.finalconv3.apply(init_weights)


    def forward(self, x):
        # Encoder
        e1 = self.features1(x)
        e2 = self.features2(e1)
        e3 = self.features3(e2)
        e4 = self.features4(e3)
        # Decoder
        d4 = self.decoder4(e4)[:,:,:e3.shape[-2],:e3.shape[-1]] + e3
        d3 = self.decoder3(d4)[:,:,:e2.shape[-2],:e2.shape[-1]] + e2
        d2 = self.decoder2(d3)[:,:,:e1.shape[-2],:e1.shape[-1]] + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)




model = Baseline()




if __name__ == "__main__":
    input = torch.rand(5, 3, 512, 512).cuda()
    model.cuda()
    print(model)
    output = model(input)
    print(output.shape)
