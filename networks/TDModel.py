## torch lib
import torch
import torch.nn as nn
import torch.nn.init as init

import functools

from utils import *
try:
    from modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')
from modules.cbam import CBAM
from modules.dual_attention import CAM_Module, TAM_Module
from modules.cc_attention import CrissCrossAttention
from . import gan_networks




class TDModel19(nn.Module):

    def __init__(self, opts, nc_in, nc_out, nc_ch):
        super(TDModel19, self).__init__()

        self.epoch = 0

        use_bias = True

        self.conv1 = nn.Conv3d(nc_in, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv5 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv8 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv10 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv11 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv14 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv16 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv17 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv18 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv19 = nn.Conv3d(nc_ch, nc_out, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.ltanh = nn.Tanh()

    def forward(self, X):

        F1 = self.relu(self.conv1(X))
        F2 = self.relu(self.conv3(self.relu(self.conv2(F1)))+F1)

        F3 = self.relu(self.conv4(F2))

        F4 = self.relu(self.conv6(self.relu(self.conv5(F3)))+F3)
        F5 = self.relu(self.conv7(F4))

        F6 = self.relu(self.conv9(self.relu(self.conv8(F5)))+F5)
        F7 = self.relu(self.conv10(F6))

        F8 = self.relu(self.conv12(self.relu(self.conv11(F7)))+F7)
        F9 = self.relu(self.conv13(F8))

        F10 = self.relu(self.conv15(self.relu(self.conv14(F9)))+F9)
        F11 = self.relu(self.conv16(F10))

        F12 = self.relu(self.conv18(self.relu(self.conv17(F11)))+F11)
        Y = self.ltanh(self.conv19(F12))

        return Y              

class TDAModel(nn.Module):

    def __init__(self, opts, nc_out, nc_ch, nf=16):
        super(TDAModel, self).__init__()

        self.epoch = 0

        self.nframes = 7
        self.center = self.nframes // 2

        use_bias = True

        ResidualBlock_noBN_begin = functools.partial(ResidualBlock_noBN, nf=nf)

        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_begin, 5)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.align = Alignment(nf=nf, groups=1)

        # self.non_local_attention = Non_Local_Attention(nf=nf, nframes=self.nframes)

        # self.temporal_fusion = Temporal_Fusion(nf=nf, nframes=self.nframes, center=self.center)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv3d_rec = TDModel(opts, 4*nf, nc_out, nc_ch)

        self.ltanh = nn.Tanh()

    def forward(self, X):

        # print(X.size())
        B, N, C, H, W = X.size()  # N video frames

        # X_max = torch.max(X, dim=2, keepdim=True)[0]
        # X_min = torch.min(X, dim=2, keepdim=True)[0]
        # X_res= X_max - X_min  
        # X_res3 = torch.cat([X_res, X_res,X_res], dim=2)
        # X[:, self.center, :, :, :]  = X_res3[:, self.center, :, :, :]
        X_res = torch.max(X, dim=2, keepdim=True)[0] - torch.min(X, dim=2, keepdim=True)[0] 
        X_4ch = torch.cat([X, X_res], dim=2)
        C = C + 1

        aligned_all_fea = []
        for b in range(B):
            x_center = X_4ch[b:b+1, self.center, :, :, :].contiguous()

            x = X_4ch[b:b+1, :, :, :, :].permute(2, 0, 1, 3, 4).view(C, N, 1, H, W)



            #### extract noisy features
            #print(x[:, :, 0, :, :].contiguous().shape)
            L1_fea_noisy = self.lrelu(self.conv_first(x[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
            L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
            # L2
            L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
            L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
            # L3
            L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
            L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

            L1_fea_noisy = L1_fea_noisy.view(C, N, -1, H, W)
            L2_fea_noisy = L2_fea_noisy.view(C, N, -1, H // 2, W // 2)
            L3_fea_noisy = L3_fea_noisy.view(C, N, -1, H // 4, W // 4)

            # ref feature list
            ref_fea_l_noisy = [
                L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
                L3_fea_noisy[:, self.center, :, :, :].clone()
            ]
            aligned_fea = []
            for i in range(N):
                if i != self.center:
                    nbr_fea_l_noisy = [
                        L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                        L3_fea_noisy[:, i, :, :, :].clone()
                    ]
                    
                    aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy).view(-1, H, W)
            
                    aligned_fea.append(aligned_fea_noisy)

            aligned_fea = torch.stack(aligned_fea, dim=1)
            
            # #non-local attention
            # non_local_feature = self.non_local_attention(aligned_noisy_fea)

            # #temporal fusion
            # fea = self.temporal_fusion(non_local_feature)# fea shape: (C, nf, H, W)     
        
            aligned_all_fea.append(aligned_fea)

        aligned_all_fea = torch.stack(aligned_all_fea, dim=0) 

        rec = self.conv3d_rec(aligned_all_fea)

        return rec     
                

class TDARevModel(nn.Module):

    def __init__(self, opts, nc_out, nc_ch, nf=16):
        super(TDARevModel, self).__init__()

        self.epoch = 0

        self.nframes = 7
        self.center = self.nframes // 2

        use_bias = True

        ResidualBlock_noBN_begin = functools.partial(ResidualBlock_noBN, nf=nf)

        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_begin, 5)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.align = Alignment(nf=nf, groups=1)

        # self.non_local_attention = Non_Local_Attention(nf=nf, nframes=self.nframes)

        # self.temporal_fusion = Temporal_Fusion(nf=nf, nframes=self.nframes, center=self.center)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv3d_rec = TDModel19(opts, 4*nf, nc_out, nc_ch)
        self.conv3d_der = gan_networks.define_G(nc_out, nc_out,  64, 'resnet_9blocks',use_dropout=False)

    def forward(self, X):

        # print(X.size())
        B, N, C, H, W = X.size()  # N video frames

        X_res = torch.max(X, dim=2, keepdim=True)[0] - torch.min(X, dim=2, keepdim=True)[0] 
        X_4ch = torch.cat([X, X_res], dim=2)
        C = C + 1

        aligned_all_fea = []
        for b in range(B):
            x_center = X_4ch[b:b+1, self.center, :, :, :].contiguous()

            x = X_4ch[b:b+1, :, :, :, :].permute(2, 0, 1, 3, 4).view(C, N, 1, H, W)



            #### extract noisy features
            #print(x[:, :, 0, :, :].contiguous().shape)
            L1_fea_noisy = self.lrelu(self.conv_first(x[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
            L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
            # L2
            L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
            L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
            # L3
            L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
            L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

            L1_fea_noisy = L1_fea_noisy.view(C, N, -1, H, W)
            L2_fea_noisy = L2_fea_noisy.view(C, N, -1, H // 2, W // 2)
            L3_fea_noisy = L3_fea_noisy.view(C, N, -1, H // 4, W // 4)

            # ref feature list
            ref_fea_l_noisy = [
                L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
                L3_fea_noisy[:, self.center, :, :, :].clone()
            ]
            aligned_fea = []
            for i in range(N):
                if i != self.center:
                    nbr_fea_l_noisy = [
                        L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                        L3_fea_noisy[:, i, :, :, :].clone()
                    ]
                    
                    aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy).view(-1, H, W)
            
                    aligned_fea.append(aligned_fea_noisy)

            aligned_fea = torch.stack(aligned_fea, dim=1)
            
            # #non-local attention
            # non_local_feature = self.non_local_attention(aligned_noisy_fea)

            # #temporal fusion
            # fea = self.temporal_fusion(non_local_feature)# fea shape: (C, nf, H, W)     
        
            aligned_all_fea.append(aligned_fea)

        aligned_all_fea = torch.stack(aligned_all_fea, dim=0) 

        rec = self.conv3d_rec(aligned_all_fea)

        return rec.view(B, C-1, H, W)    

    def forward_derain(self, X):

        # print(X.size())
        B, N, C, H, W = X.size()  # N video frames

        X_res = torch.max(X, dim=2, keepdim=True)[0] - torch.min(X, dim=2, keepdim=True)[0] 
        X_4ch = torch.cat([X, X_res], dim=2)
        C = C + 1

        aligned_all_fea = []
        for b in range(B):
            x_center = X_4ch[b:b+1, self.center, :, :, :].contiguous()

            x = X_4ch[b:b+1, :, :, :, :].permute(2, 0, 1, 3, 4).view(C, N, 1, H, W)



            #### extract noisy features
            #print(x[:, :, 0, :, :].contiguous().shape)
            L1_fea_noisy = self.lrelu(self.conv_first(x[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
            L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
            # L2
            L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
            L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
            # L3
            L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
            L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

            L1_fea_noisy = L1_fea_noisy.view(C, N, -1, H, W)
            L2_fea_noisy = L2_fea_noisy.view(C, N, -1, H // 2, W // 2)
            L3_fea_noisy = L3_fea_noisy.view(C, N, -1, H // 4, W // 4)

            # ref feature list
            ref_fea_l_noisy = [
                L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
                L3_fea_noisy[:, self.center, :, :, :].clone()
            ]
            aligned_fea = []
            for i in range(N):
                if i != self.center:
                    nbr_fea_l_noisy = [
                        L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                        L3_fea_noisy[:, i, :, :, :].clone()
                    ]
                    
                    aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy).view(-1, H, W)
            
                    aligned_fea.append(aligned_fea_noisy)

            aligned_fea = torch.stack(aligned_fea, dim=1)
            
            # #non-local attention
            # non_local_feature = self.non_local_attention(aligned_noisy_fea)

            # #temporal fusion
            # fea = self.temporal_fusion(non_local_feature)# fea shape: (C, nf, H, W)     
        
            aligned_all_fea.append(aligned_fea)

        aligned_all_fea = torch.stack(aligned_all_fea, dim=0) 

        rec = self.conv3d_rec(aligned_all_fea)
        Y = self.conv3d_der(rec.view(B, C-1, H, W))

        return Y

    def forward_both(self, X):

        # print(X.size())
        B, N, C, H, W = X.size()  # N video frames

        X_res = torch.max(X, dim=2, keepdim=True)[0] - torch.min(X, dim=2, keepdim=True)[0] 
        X_4ch = torch.cat([X, X_res], dim=2)
        C = C + 1

        aligned_all_fea = []
        for b in range(B):
            x_center = X_4ch[b:b+1, self.center, :, :, :].contiguous()

            x = X_4ch[b:b+1, :, :, :, :].permute(2, 0, 1, 3, 4).view(C, N, 1, H, W)



            #### extract noisy features
            #print(x[:, :, 0, :, :].contiguous().shape)
            L1_fea_noisy = self.lrelu(self.conv_first(x[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
            L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
            # L2
            L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
            L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
            # L3
            L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
            L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

            L1_fea_noisy = L1_fea_noisy.view(C, N, -1, H, W)
            L2_fea_noisy = L2_fea_noisy.view(C, N, -1, H // 2, W // 2)
            L3_fea_noisy = L3_fea_noisy.view(C, N, -1, H // 4, W // 4)

            # ref feature list
            ref_fea_l_noisy = [
                L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
                L3_fea_noisy[:, self.center, :, :, :].clone()
            ]
            aligned_fea = []
            for i in range(N):
                if i != self.center:
                    nbr_fea_l_noisy = [
                        L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                        L3_fea_noisy[:, i, :, :, :].clone()
                    ]
                    
                    aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy).view(-1, H, W)
            
                    aligned_fea.append(aligned_fea_noisy)

            aligned_fea = torch.stack(aligned_fea, dim=1)
            
            # #non-local attention
            # non_local_feature = self.non_local_attention(aligned_noisy_fea)

            # #temporal fusion
            # fea = self.temporal_fusion(non_local_feature)# fea shape: (C, nf, H, W)     
        
            aligned_all_fea.append(aligned_fea)

        aligned_all_fea = torch.stack(aligned_all_fea, dim=0) 

        rec = self.conv3d_rec(aligned_all_fea)

        Y = self.conv3d_der(rec.view(B, C-1, H, W))

        return rec.view(B, C-1, H, W), Y
                 

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out
class Alignment(nn.Module):

    def __init__(self, nf=64, groups=1):
        super(Alignment, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                   deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l_noisy, ref_fea_l_noisy):
        '''align other neighboring frames to the reference frame in the feature level
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l_noisy[2], ref_fea_l_noisy[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        # L3_fea_predenoised = self.lrelu(self.L3_dcnpack(nbr_fea_l_noisy[2], L3_offset))
        L3_fea_noisy = self.lrelu(self.L3_dcnpack(nbr_fea_l_noisy[2], L3_offset))
        # L2
        L2_offset = torch.cat([nbr_fea_l_noisy[1], ref_fea_l_noisy[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea_noisy = self.L2_dcnpack(nbr_fea_l_noisy[1], L2_offset)
        L3_fea_noisy = F.interpolate(L3_fea_noisy, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea_noisy = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea_noisy, L3_fea_noisy], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l_noisy[0], ref_fea_l_noisy[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea_noisy = self.L1_dcnpack(nbr_fea_l_noisy[0], L1_offset)
        L2_fea_noisy = F.interpolate(L2_fea_noisy, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea_noisy = self.L1_fea_conv(torch.cat([L1_fea_noisy, L2_fea_noisy], dim=1))
        # Cascading
        offset = torch.cat([L1_fea_noisy, ref_fea_l_noisy[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea_noisy = self.lrelu(self.cas_dcnpack(L1_fea_noisy, offset))

        return L1_fea_noisy

class Non_Local_Attention(nn.Module):

    def __init__(self, nf=64, nframes=3):
        super(Non_Local_Attention, self).__init__()

        self.conv_before_cca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())      
        self.conv_before_ca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_before_ta = nn.Sequential(nn.Conv2d(nframes, nframes, 3, padding=1, bias=False),
                                   nn.ReLU())

        self.recurrence = 2
        self.cca = CrissCrossAttention(nf)
        self.ca = CAM_Module()
        self.ta = TAM_Module()

        self.conv_after_cca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_after_ca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_after_ta = nn.Sequential(nn.Conv2d(nframes, nframes, 3, padding=1, bias=False),
                                   nn.ReLU())

        self.conv_final = nn.Conv2d(nf, nf, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  

        # spatial non-local attention
        cca_feat = self.conv_before_cca(aligned_fea.reshape(-1, C, H, W))
        for i in range(self.recurrence):
            cca_feat = self.cca(cca_feat)
        cca_conv = self.conv_after_cca(cca_feat).reshape(B, N, C, H, W)

        # channel non-local attention
        ca_feat = self.conv_before_ca(aligned_fea.reshape(-1, C, H, W))
        ca_feat = self.ca(ca_feat)
        ca_conv = self.conv_after_ca(ca_feat).reshape(B, N, C, H, W)

        # temporal non-local attention
        ta_feat = self.conv_before_ta(aligned_fea.permute(0, 2, 1, 3, 4).reshape(-1, N, H, W))
        ta_feat = self.ta(ta_feat)
        ta_conv = self.conv_after_ta(ta_feat).reshape(B, C, N, H, W).permute(0, 2, 1, 3, 4)

        feat_sum = cca_conv+ca_conv+ta_conv
        
        output = self.conv_final(feat_sum.reshape(-1, C, H, W)).reshape(B, N, C, H, W)
                
        return aligned_fea + output


class Temporal_Fusion(nn.Module):

    def __init__(self, nf=64, nframes=3, center=1):
        super(Temporal_Fusion, self).__init__()
        self.center = center

        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nonlocal_fea):
        B, N, C, H, W = nonlocal_fea.size()  

        emb_ref = self.tAtt_2(nonlocal_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(nonlocal_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1) 
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1)
        cor_prob = cor_prob.view(B, -1, H, W)
        nonlocal_fea = nonlocal_fea.view(B, -1, H, W) * cor_prob

        fea = self.lrelu(self.fea_fusion(nonlocal_fea))

        att = self.lrelu(self.sAtt_1(nonlocal_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))

        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add

        return fea