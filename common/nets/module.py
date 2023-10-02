import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from nets.layer import make_conv_layers, make_conv1d_layers, make_deconv_layers, make_linear_layers
from utils.human_models import mano
from utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from config import cfg
from nets.crosstransformer import CrossTransformer
from einops import rearrange
from timm.models.vision_transformer import Block

class Transformer(nn.Module):
    def __init__(self, in_chans=512, joint_num=21, depth=4, num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, joint_num, in_chans))
        self.blocks = nn.ModuleList([
            Block(in_chans, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for i in range(depth)])
    def forward(self, x):
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x

class FuseFormer(nn.Module):
    def __init__(self):
        super(FuseFormer, self).__init__()
        self.FC = nn.Linear(512*2, 512)
        self.pos_embed = nn.Parameter(torch.randn(1, 1+(2*8*8), 512))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.SA_T = nn.ModuleList([
            Block(512, 4, 4.0, qkv_bias=False, norm_layer=nn.LayerNorm)
            for i in range(4)])
        self.FC2 = nn.Linear(512, 512)
        #Decoder
        self.CA_T = CrossTransformer()
        self.FC3 = nn.Linear(512, 512)

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        feat1 = rearrange(feat1, 'B C H W -> B (H W) C')
        feat2 = rearrange(feat2, 'B C H W -> B (H W) C')
        # joint Token
        token_j = self.FC(torch.cat((feat1, feat2), dim=-1))
        
        # similar token
        token_s = torch.cat((feat1, feat2), dim=1) + self.pos_embed[:,1:]
        cls_token = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        token_s = torch.cat((cls_token, token_s), dim=1)
        for blk in self.SA_T:
            token_s = blk(token_s)
        token_s = self.FC2(token_s)

        output = self.CA_T(token_j, token_s)
        output = self.FC3(output)
        output = rearrange(output, 'B (H W) C -> B C H W', H=H, W=W)
        return output



class EABlock(nn.Module):
    def __init__(self):
        super(EABlock, self).__init__()
        self.conv_l = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.conv_r = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.Extract = FuseFormer()
        self.Adapt_r = FuseFormer()
        self.Adapt_l = FuseFormer()
        self.conv_l2 = make_conv_layers([512*2, 512*2], kernel=1, stride=1, padding=0)
        self.conv_r2 = make_conv_layers([512*2, 512*2], kernel=1, stride=1, padding=0)

    def forward(self, hand_feat):
        rhand_feat = self.conv_r(hand_feat)
        lhand_feat = self.conv_l(hand_feat)
        inter_feat = self.Extract(rhand_feat, lhand_feat)
        rinter_feat = self.Adapt_r(rhand_feat, inter_feat)
        linter_feat = self.Adapt_l(lhand_feat, inter_feat)
        rhand_feat = self.conv_r2(torch.cat((rhand_feat,rinter_feat),dim=1))
        lhand_feat = self.conv_l2(torch.cat((lhand_feat,linter_feat),dim=1))
        return rhand_feat, lhand_feat



class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.EABlock = EABlock()
        self.conv_r2 = make_conv_layers([512*2, self.joint_num*cfg.output_hm_shape[2]], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_l2 = make_conv_layers([512*2, self.joint_num*cfg.output_hm_shape[2]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, hand_feat):
        rhand_feat, lhand_feat = self.EABlock(hand_feat)
        rhand_hm = self.conv_r2(rhand_feat)
        rhand_hm = rhand_hm.view(-1,self.joint_num, cfg.output_hm_shape[2], cfg.output_hm_shape[0], cfg.output_hm_shape[1])
        rhand_coord = soft_argmax_3d(rhand_hm)

        lhand_hm = self.conv_l2(lhand_feat)
        lhand_hm = lhand_hm.view(-1,self.joint_num, cfg.output_hm_shape[2], cfg.output_hm_shape[0], cfg.output_hm_shape[1])
        lhand_coord = soft_argmax_3d(lhand_hm)

        return rhand_coord, lhand_coord, rhand_feat, lhand_feat



class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.rconv = make_conv_layers([1024,512], kernel=1, stride=1, padding=0)
        self.lconv = make_conv_layers([1024,512], kernel=1, stride=1, padding=0)
        self.rshape_out = make_linear_layers([1024, mano.shape_param_dim], relu_final=False)
        self.rcam_out = make_linear_layers([1024, 3], relu_final=False)
        self.lshape_out = make_linear_layers([1024, mano.shape_param_dim], relu_final=False)
        self.lcam_out = make_linear_layers([1024, 3], relu_final=False)
        #SJT
        self.Transformer_r = Transformer(in_chans=512, joint_num=21, depth=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm)
        self.Transformer_l = Transformer(in_chans=512, joint_num=21, depth=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm)
        #relative translation
        self.root_relative = make_linear_layers([2*(1024),512,3], relu_final=False)
        ##
        self.rroot_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
        self.rpose_out = make_linear_layers([self.joint_num*(512+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint
        self.lroot_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
        self.lpose_out = make_linear_layers([self.joint_num*(512+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint

    def forward(self, rhand_feat, lhand_feat, rjoint_img, ljoint_img):
        batch_size = rhand_feat.shape[0]

        # shape and camera parameters
        rshape_param = self.rshape_out(rhand_feat.mean((2,3)))
        rcam_param = self.rcam_out(rhand_feat.mean((2,3)))
        lshape_param = self.lshape_out(lhand_feat.mean((2,3)))
        lcam_param = self.lcam_out(lhand_feat.mean((2,3)))
        rel_trans = self.root_relative(torch.cat((rhand_feat, lhand_feat), dim=1).mean((2,3)))

        # xyz corrdinate feature
        rhand_feat = self.rconv(rhand_feat)
        lhand_feat = self.lconv(lhand_feat)
        rhand_feat = sample_joint_features(rhand_feat, rjoint_img[:,:,:2]) # batch_size, joint_num, feat_dim
        lhand_feat = sample_joint_features(lhand_feat, ljoint_img[:,:,:2]) # batch_size, joint_num, feat_dim

        # import pdb; pdb.set_trace()
        rhand_feat = self.Transformer_r(rhand_feat)
        lhand_feat = self.Transformer_l(lhand_feat)

        # Relative Translation
        rhand_feat = torch.cat((rhand_feat, rjoint_img),2).view(batch_size,-1)
        lhand_feat = torch.cat((lhand_feat, ljoint_img),2).view(batch_size,-1)

        rroot_pose = self.rroot_pose_out(rhand_feat)
        rpose_param = self.rpose_out(rhand_feat)
        lroot_pose = self.lroot_pose_out(lhand_feat)
        lpose_param = self.lpose_out(lhand_feat)

        return rroot_pose, rpose_param, rshape_param, rcam_param, lroot_pose, lpose_param, lshape_param, lcam_param, rel_trans

