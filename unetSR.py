# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:47:18 2020


ejemplo de clase sin normalización por lotes

@author: villacuPC
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

#%% BLOQUES DE UNET
class doubleconv2d(nn.Module):
  def __init__(self,in_ch,out_ch):#no hay canales intermedios, en el ejemplo se muestran las modificaciones necesarias
    super().__init__()
    self.doubleconv=nn.Sequential(
        nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self,x):
    return self.doubleconv(x)

#class inconv DE EJEMPLO DEL PROFE
class inconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(inconv,self).__init__()
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = doubleconv2d(in_ch,out_ch)
    def forward(self,x):
        x=self.up(x)
        return self.conv(x)
    

class Down(nn.Module):
    #maxpool->doubleconv
    def __init__(self,in_ch,out_ch):
        super().__init__()
        #con maxpool al inicio
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            doubleconv2d(in_ch,out_ch)
            )
    def forward(self,x):
        return self.down(x)
    

class Up(nn.Module):
  #upsample-> concatenate ->doubleconv
  def __init__(self,in_ch,out_ch):
    super().__init__()
    self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
    self.conv = doubleconv2d(in_ch,out_ch)#entender que es esto
    
  def forward(self,x1,x2):
    x1 = self.up(x1)
    
    # input is CHW PONER SI NO FUNCIONA medida de seguridad¿?
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    
    x = torch.cat([x2, x1], dim=1) 
    return self.conv(x)

class conv_out(nn.Module): #DIFERENTE AL DEL PROFE
    def __init__(self, in_ch, out_ch):
        super(conv_out, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
#%% clase UNET
class UNet(nn.Module):
  def __init__(self,in_ch,out_ch):
    super(UNet,self).__init__()
    self.in_channels = in_ch
    self.out_channels = out_ch
    #capa de entrada
    self.in_layer = inconv(in_ch,64)
    self.down1 = Down(64, 128)
    self.down2 = Down(128, 256)
    self.down3 = Down(256, 512)
    self.down4 = Down(512, 512)
    self.up1 = Up(1024, 256)
    self.up2 = Up(512, 128)
    self.up3 = Up(256, 64)
    self.up4 = Up(128, 64)
    self.out_layer = conv_out(64, out_ch)
  def forward(self,x):
    x1 = self.in_layer(x) #16
    x2 = self.down1(x1)#8
    x3 = self.down2(x2)#4
    x4 = self.down3(x3)#2
    
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    
    x = self.up3(x, x2)#8
    x = self.up4(x, x1)#16
    out = self.out_layer(x)
    return out