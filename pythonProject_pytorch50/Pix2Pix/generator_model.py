import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") if down else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Block_down(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block_down, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 6, 2, 1, bias=False, padding_mode="reflect") if down else nn.ConvTranspose2d(in_channels, out_channels, 6, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Block_up(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block_up, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 6, 2, 1, bias=False, padding_mode="reflect") if down else nn.ConvTranspose2d(in_channels, out_channels, 6, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super().__init__()

        self.initial_down_1 = nn.Sequential(
            nn.Conv2d(10, 50, 1, 1, 0, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )   # N*50*1*31
        self.initial_down_2 = nn.Sequential(
            nn.Conv2d(31, 50, 1, 1, 0, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )   # N*50*1*50

        self.initial_down = nn.Sequential(
            nn.Conv2d(1, features, 3, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )   # 48

        self.down1 = Block_down(features, features * 2, down=True, act="leaky", use_dropout=False) # 24
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        ) # 12
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )  # 6
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        ) # 3
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 3, 1, 1), nn.ReLU()
        )  # 3

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block_up(
            features * 2 * 2, features * 1, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down_1(x)   # N*10*1*31--->N*50*1*31
        d1 = self.initial_down_2(d1.transpose(3, 1))   # N*50*1*31--->N*50*1*50
        d1 = d1.reshape(-1, 1, 50, 50)

        d1 = self.initial_down(d1)   # N*64*50*50

        d2 = self.down1(d1)  # N*128*24*24
        d3 = self.down2(d2)  # N*256*12*12
        d4 = self.down3(d3)  # N*512*6*6
        d5 = self.down4(d4)  # N*512*3*3
        # d6 = self.down5(d5)
        # d7 = self.down6(d6)
        bottleneck = self.bottleneck(d5)  # N*512*3*3
        up1 = self.up1(bottleneck)  # N*512*6*6
        up2 = self.up2(torch.cat([up1, d4], 1))  # N*256*12*12
        up3 = self.up3(torch.cat([up2, d3], 1))  # N*128*24*24
        up4 = self.up4(torch.cat([up3, d2], 1))  # N*64*50*50
        # up5 = self.up5(torch.cat([up4, d4], 1))
        # up6 = self.up6(torch.cat([up5, d3], 1))
        # up7 = self.up7(torch.cat([up6, d2], 1))

        output = self.final_up(torch.cat([up4, d1], 1))
        return output


def test():
    x = torch.randn((2, 10, 1, 31))
    model = Generator(in_channels=1, features=32)
    preds = model(x)
    print(preds.shape)   # 1*1*50*50


if __name__ == "__main__":
    test()
