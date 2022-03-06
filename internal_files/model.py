from . import config
import torch.nn as nn
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

def ret(x): return x

class Block(nn.Module):
    """
    Create a unet block that will take the number of input and output channels, and apply two convolutions to give this number of channels
    """
    def __init__(self, in_ch, out_ch, ks=3, stride=1, normalize=False, pad=False):
        super().__init__()
        padding = ks//2 if pad else 0
        self.conv1 = nn.Conv2d(in_ch, out_ch, ks, stride, padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, ks, stride, padding)
        self.normalize = nn.BatchNorm2d(out_ch) if normalize else ret
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # apply the block to some input
        return self.conv2(self.relu(self.normalize((self.conv1(x)))))


class Encoder(nn.Module):
    def __init__(self, channels=[3, 16, 32, 64]):
        super().__init__()
        # create 3 encoder blocks according to the channels 
        self.enc_blocks = nn.ModuleList([
            Block(channels[i], channels[i+1]) for i in range(len(channels)-1)
        ])
        # create max pooling 2x2 layer
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x): 
        enc_outs = []
        # loop through encoder blocks and store results in list
        for block in self.enc_blocks:
            x = block(x)
            enc_outs.append(x)
            # apply pooling before going to the next block
            x = self.pool(x)
        # return list
        return enc_outs


class Decoder(nn.Module):
    def __init__(self, channels=[64, 32, 16]):
        super().__init__()
        # create upsampling layers
        self.up_layers = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels) - 1)
        ])
        # create the decoder blocks to use after the upsampling layers 
        self.dec_blocks = nn.ModuleList([
            Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)
        ])

    def crop(self, x, encoder_x):
        """
        Crop the encoder x so that the size matches the current upsample x size
        """
        B, C, H, W = x.shape
        # create a centre crop using the dimensions of the current x on the encoder layer output
        centre_crop = CenterCrop((H, W))
        encoder_x = centre_crop(encoder_x)
        return encoder_x
    
    def forward(self, x, encoder_list): 
        for up_layer, enc_x, block in zip(self.up_layers, encoder_list, self.dec_blocks):
            # the encoder is doing encoder block -> pooling so the decoder must do upsampling -> concat -> decoder block for sizes on concat to match
            x = up_layer(x)
            # match up the encoder feature sizes with upsampled decoder size
            enc_x = self.crop(x, enc_x)
            # concat x and enc_x along the channel axis
            # if x.shape = (B,C,H,W) = (64, 32, 16, 16) = enc_x.shape => torch.cat([x, enc_x], axis=1).shape = (64, 64, 16, 16) as required 
            x = torch.cat([x, enc_x], axis=1)
            # the decoder block will take high level feature channel information + lower level localization information and reduce this to a lower channel space
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(self, out_channels, enc_channels=[3, 16, 32, 64], dec_channels=[64, 32, 16], keep_dim=True, output_size=(config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)):
        """
        Define a UNet to output the probabilites of some class (each output channel is a class) using encoder and decoder modules
        """
        super().__init__()
        # define the encoder and decoder
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        # compress decoder activation channels to the number of classes
        self.class_layer = nn.Conv2d(dec_channels[-1], out_channels, 1)
        self.keep_dim = keep_dim
        self.output_size = output_size
    
    def forward(self, x): 
        # pass the input to the encoder to create an encoder list of activation layers
        enc_list = self.encoder(x)
        rev_enc_list = enc_list[::-1]

        # pass the decoder final layer of the encoder as well as all other intermediate layers
        dec_x = self.decoder(
            rev_enc_list[0], # final layer of the encoder
            rev_enc_list[1:] # all other layers of the encoder from 'bottom' to 'top'
        )

        # apply the convolution to map to a number of feature classes
        class_maps = self.class_layer(dec_x)

        # use pytorch functional interpolate to upsample and input tensor to a desired size if keep_dim == True
        # TODO: investigate the effects of upsampling by a large amount using this function (is there a better way?)
        if self.keep_dim == True:
            # note that size is the output spatial size - does not accout for batches or channels
            class_maps = F.interpolate(class_maps, self.output_size)
        
        return class_maps

