import torch.nn as nn
from src.utils.utils import normal_init

class Encoder_Block(nn.Module):
    def __init__(self,in_channels,out_channels,normalize=True,activation="leaky"):
        super(Encoder_Block,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=4,
                              stride=2,
                              padding=1,
                              bias=False,
                              padding_mode="reflect")
        self.normalize=normalize
        if self.normalize:
            self.norm = nn.BatchNorm2d(out_channels)
        if activation=="leaky":
            self.act = nn.LeakyReLU(0.2)
        elif activation=="relu":
            self.act = nn.ReLU()
            
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
    def forward(self,x):
        x=self.conv(x)
        if self.normalize:
            x=self.norm(x)
        x=self.act(x)
        return x

class Decoder_Block(nn.Module):
    
    def __init__(self,in_channels,out_channels,normalize=True,dropout=True,activation=True,bias=False):
        super(Decoder_Block,self).__init__()
        self.conv_trans=nn.ConvTranspose2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           bias=bias)
        self.normalize = normalize
        self.activation = activation
        if self.normalize:
            self.norm=nn.BatchNorm2d(out_channels)
        self.drop = dropout
        if self.drop:
            self.dropout=nn.Dropout(p=0.5)
        self.relu=nn.ReLU()
    
    def weight_init(self,mean,std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
  
    def forward(self,x):
        x=self.conv_trans(x)
        if self.normalize:
            x=self.norm(x)
        if self.activation:
            x=self.relu(x)
        if self.drop:
            x=self.dropout(x)
        return x