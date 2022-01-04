import torch
import torch.nn as nn
from src.models.modules.generator import Encoder_Block, Decoder_Block

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.encoder_block1=Encoder_Block(in_channels=3,
                                          out_channels=64,
                                          normalize=False)
        #self.encoder_block1.weight_init(mean=0.0, std=0.02)
        
        self.encoder_block2=Encoder_Block(in_channels=64,
                                          out_channels=128)
        #self.encoder_block2.weight_init(mean=0.0, std=0.02)
        
        self.encoder_block3=Encoder_Block(in_channels=128,
                                          out_channels=256)
        #self.encoder_block3.weight_init(mean=0.0, std=0.02)
        
        self.encoder_block4=Encoder_Block(in_channels=256,
                                          out_channels=512)
        #self.encoder_block4.weight_init(mean=0.0, std=0.02)
        
        self.encoder_block5=Encoder_Block(in_channels=512,
                                          out_channels=512)
        #self.encoder_block5.weight_init(mean=0.0, std=0.02)
        
        self.encoder_block6=Encoder_Block(in_channels=512,
                                          out_channels=512)
        #self.encoder_block6.weight_init(mean=0.0, std=0.02)
        
        self.encoder_block7=Encoder_Block(in_channels=512,
                                          out_channels=512)
        #self.encoder_block7.weight_init(mean=0.0, std=0.02)
        
        self.encoder_block8=Encoder_Block(in_channels=512,
                                          out_channels=512,
                                          normalize=False,
                                          activation="relu")
        #self.encoder_block8.weight_init(mean=0.0, std=0.02)
        
    def forward(self,x):
        x1=self.encoder_block1(x)
        x2=self.encoder_block2(x1)
        x3=self.encoder_block3(x2)
        x4=self.encoder_block4(x3)
        x5=self.encoder_block5(x4)
        x6=self.encoder_block6(x5)
        x7=self.encoder_block7(x6)
        x8=self.encoder_block8(x7)
        return x1,x2,x3,x4,x5,x6,x7,x8


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.encoder = Encoder()
        self.decoder_block1 = Decoder_Block(in_channels=512,
                                            out_channels=512)
        #self.decoder_block1.weight_init(mean=0.0, std=0.02)
        
        self.decoder_block2 = Decoder_Block(in_channels=1024,
                                            out_channels=512)
        #self.decoder_block2.weight_init(mean=0.0, std=0.02)
        
        self.decoder_block3 = Decoder_Block(in_channels=1024,
                                            out_channels=512)
        #self.decoder_block3.weight_init(mean=0.0, std=0.02)
        
        self.decoder_block4 = Decoder_Block(in_channels=1024,
                                            out_channels=512,
                                            dropout=False)
        #self.decoder_block4.weight_init(mean=0.0, std=0.02)
        
        self.decoder_block5 = Decoder_Block(in_channels=1024,
                                            out_channels=256,
                                            dropout=False)
        #self.decoder_block5.weight_init(mean=0.0, std=0.02)
        
        self.decoder_block6 = Decoder_Block(in_channels=512,
                                            out_channels=128,
                                            dropout=False)
        #self.decoder_block6.weight_init(mean=0.0, std=0.02)
        
        self.decoder_block7 = Decoder_Block(in_channels=256,
                                            out_channels=64,
                                            dropout=False)
        #self.decoder_block7.weight_init(mean=0.0, std=0.02)
        
        self.decoder_block8 = Decoder_Block(in_channels=128,
                                            out_channels=3,
                                            normalize=False,
                                            dropout=False,
                                            activation=False,
                                            bias=True)
        #self.decoder_block8.weight_init(mean=0.0, std=0.02)
        
    def forward(self,x):
        x1,x2,x3,x4,x5,x6,x7,x8 = self.encoder(x)
        x = self.decoder_block1(x8)
        x = torch.cat([x7,x],dim=1)
        x = self.decoder_block2(x)
        x = torch.cat([x6,x],dim=1)
        x = self.decoder_block3(x)
        x = torch.cat([x5,x],dim=1)
        x = self.decoder_block4(x)
        x = torch.cat([x4,x],dim=1)
        x = self.decoder_block5(x)
        x = torch.cat([x3,x],dim=1)
        x = self.decoder_block6(x)
        x = torch.cat([x2,x],dim=1)
        x = self.decoder_block7(x)
        x = torch.cat([x1,x],dim=1)
        x = self.decoder_block8(x)
        out = torch.tanh(x)
        return out
        