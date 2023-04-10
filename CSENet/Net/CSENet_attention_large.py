import torch
import torch.nn as nn
import torchvision.models

bachbone = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.DEFAULT)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel*reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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




class Connectivity(nn.Module):
    def __init__(self, num_classes=1, num_neighbor=8):
        super(Connectivity, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        filters = (192, 384, 768, 1536)
        # self.weights = nn.Parameter(torch.ones(4, requires_grad=True))
        self.features1 = bachbone.features[:2]
        self.features2 = bachbone.features[2:4]
        self.features3 = bachbone.features[4:6]
        self.features4 = bachbone.features[6:]

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 2)
        self.finalrelu1 = nn.ReLU()
        self.finalconv2 = nn.Conv2d(32, 64, 3)
        self.finalrelu2 = nn.ReLU()  # 共享分割头

        # self.angle_conv3 = nn.Conv2d(64, 38, 3, 1, 1)
        self.sobel = nn.Conv2d(64, num_classes, 3, 1, 1)
        self.finalconv0 = nn.Conv2d(64, out_channels=num_classes, kernel_size=1)

        self.finalconvd1 = nn.Conv2d(64, out_channels=num_neighbor, kernel_size=3, dilation=2, padding=2)
        self.finalse1 = SELayer(8)

        self.finalconvd2 = nn.Conv2d(64, out_channels=num_neighbor, kernel_size=3, dilation=3, padding=3)
        self.finalse2 = SELayer(8)

        self.unfold = nn.Unfold(kernel_size=10, stride=10)
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=18, nhead=2, dim_feedforward=64, batch_first=True)
        self.fold = nn.Fold(output_size=(500,500), kernel_size=10, stride=10)

        self.mlp = nn.Sequential(
            nn.Conv2d(18,1,1,1,0))





        self.decoder4.apply(init_weights)
        self.decoder3.apply(init_weights)
        self.decoder2.apply(init_weights)
        self.decoder1.apply(init_weights)
        self.finaldeconv1.apply(init_weights)
        self.finalconv2.apply(init_weights)
        self.finalconv0.apply(init_weights)
        self.finalconvd1.apply(init_weights)
        self.finalse1.apply(init_weights)
        self.finalconvd2.apply(init_weights)
        self.finalse2.apply(init_weights)
        # self.angle_conv3.apply(init_weights)
        self.sobel.apply(init_weights)
        self.mlp.apply(init_weights)




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
        out = self.finalrelu2(out)   # 共享分割头

        out0 = self.finalconv0(out)
        sobel = self.sobel(out)
        # angle = self.angle_conv3(out)


        outd1 = self.finalconvd1(out) - out0
        outd1 = self.finalse1(outd1)+outd1

        outd2 = self.finalconvd2(out) - out0
        outd2 = self.finalse2(outd2)+outd2

        out = torch.cat([out0, outd1, outd2, sobel], 1)

        n, c, h, w = out.shape
        out = out.unsqueeze(2).reshape(n * c, 1, h, w)
        out = self.unfold(out)
        _, h1, w1 = out.shape
        out = out.reshape(n, c, h1, w1).permute(0, 3, 2, 1).reshape(n * w1, h1, c)
        out = self.encoderlayer(out)
        out = out.reshape(n, w1, h1, c).permute(0, 3, 2, 1).reshape(_, h1, w1)
        out = self.fold(out).reshape(n, c, h, w)

        out = self.mlp(out) + out0



        return torch.sigmoid(out), torch.sigmoid(outd1), torch.sigmoid(outd2), torch.sigmoid(sobel)


model =  Connectivity(num_classes=1,num_neighbor=8)   #  改造




if __name__ == "__main__":
    input = torch.rand(2, 3, 500, 500).cuda()
    model.cuda()
    print(model)
    output = model(input)
    print([x.shape for x in output])
