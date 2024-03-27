import torch.nn as nn
import torch


class GRU_CNNNet_Large(nn.Module):
    def __init__(self, t=13, n_well=6, h=180, w=140, hidden_size=32):
        super(GRU_CNNNet_Large, self).__init__()
        self.gru = nn.GRU(input_size=n_well, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size * 2, out_channels=hidden_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU()
        )

        self.upsample1 = nn.Upsample(size=(30, 14), mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(size=(180, 140), mode='bilinear', align_corners=False)
        nn.ConvTranspose2d

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size * 2 + 3, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU()
        )
        self.cnn2 = nn.Conv2d(in_channels=hidden_size, out_channels=t, kernel_size=1)
        self.h = h
        self.w = w
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x, x0):
        print(x.shape)
        x, _ = self.gru(x)
        print(x.shape)
        x = x.unsqueeze(-1)
        print(x.shape)
        x = x.transpose(-2, -3)
        # print(x.shape)
        # x = torch.repeat_interleave(x.unsqueeze(2), repeats=self.w, dim=2)
        x = self.upsample1(x)
        # print(x.shape)
        x = self.upsample2(x)
        print(x.shape)
        # x = self.cnn(x.transpose(-1, -2).transpose(-2, -3))
        # x = x.reshape(self.h, self.w)\
        x = self.cnn(x)
        print(x.shape)
        x = self.dropout(x)
        # x = self.upsample1(x)
        print(x.shape)
        print(x0.shape)
        #         x0 = torch.repeat_interleave(x0, repeats=x.size(0), dim=0)
        # print(x.shape)

        x = torch.cat((x0, x), 1)

        #         x = self.dropout(x)
        x = self.cnn1(x)

        x = self.cnn2(x)
        print(x.shape)
        shape_x = x.shape
        return x

#         return torch.sigmoid(x)
#         return torch.softmax(x.view(shape_x[0], shape_x[1], -1), dim=-1).view(shape_x)
