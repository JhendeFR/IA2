import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

#Simplemente aplila varios bloques en secuencia
def make_layer(block, n_layers): #algun tipo de bloque (RRDB o ResidualDenseBlock_5C) y el numero de capas
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True): #nf: numero de canales de entrada, gc: cada cnn añade 32 canales y bias: para el sesgo
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias) #recibe nf canales y genera tensor con gc canales (ventana 3x3, 1 stride,1 padding)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias) #recibe la conv anterior y produce otros gc canales
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True) #Variante de ReLU que permite el paso de valores negativos

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x)) #=32
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1))) #=128
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1))) #=160
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1))) #=192
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) # concatena todos los canales y los reduce a nf
        return x5 * 0.2 + x #tensor concatenado escalado y tensor de entrada (residual connection)


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

# Modelo PSNRNet basado en RRDBNet
class PSNRNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32): #nb: numero de bloques RRDB = 23 ejemplo
        super(PSNRNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc) #fija nf y gc para crear bloques RRDB

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True) #cabezal
        self.RRDB_trunk = make_layer(RRDB_block_f, nb) #apila nb bloques RRDB
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x) #extraccion inicial
        trunk = self.trunk_conv(self.RRDB_trunk(fea)) #pasa por los bloques RRDB
        fea = fea + trunk #residual global(extraccion inicial + salida de bloques enriquecida) CLAVE PARA PSNR
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))) #upsampling x2
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))) #upsampling x2
        out = self.conv_last(self.lrelu(self.HRconv(fea))) #reconstruccion final
        return out
