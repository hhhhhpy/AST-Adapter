from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
import math
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(size=shape,requires_grad = False)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=-1)

class GumbelNetwork(nn.Module):
    def __init__(self, pre_network, sub_networks, post_network, predictor, tau=1.0):
        super().__init__()

        self.pre_network  = pre_network
        self.sub_nets     = nn.ModuleList(sub_networks)
        self.post_network = post_network
        self.predictor = predictor

        self.tau = tau
    
    def forward(self, x):
        logits = self.predictor(x)
        
        prep_x = self.pre_network(x)
        values = [m(prep_x) for m in self.sub_nets]
        # print('training:{}'.format(self.training))
        if self.training:

            logp = F.log_softmax(logits * (1/self.tau), dim=-1)
            gumb = F.gumbel_softmax(logp, tau=self.tau, hard=True)

            val = torch.stack(values, dim=1) # b,k,...
            out = torch.einsum("bk,bk...->b...", gumb, val)

            idx = torch.max(gumb, dim=-1)[1]
        else:
            logp = F.log_softmax(logits * (1/self.tau), dim=-1)
            gumb = F.gumbel_softmax(logp, tau=self.tau, hard=True)

            val = torch.stack(values, dim=1) # b,k,...

            idx = torch.max(logits, dim=-1)[1]
            gbl = F.one_hot(idx, val.size(1)).float()
            out = torch.einsum("bk,bk...->b...", gbl, val)
        
        out = self.post_network(out)

        aux = {
            "index": idx,
            "log_p": logp,
            "gumbl": gumb
        }
        return out, aux
    
class paramnetwork(nn.Module):
    def __init__(self,t):
        super().__init__()
        self.p = nn.Parameter(t)
    def forward(self,x):
        return self.p.unsqueeze(0).expand(x.size(0),-1)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.p = attn_drop
        # self.log = None
    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

      
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            x = F.scaled_dot_product_attention(q,k,v,dropout_p=self.p).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None,config=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.config = config
        
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
        
        def make_down():
            l = nn.Conv3d(dim, config.bottle_dim, (1,1,1), (1,1,1), (0,0,0))
            if config.dataset =='HMDB51' or self.config.dataset=='UCF_101' or self.config.dataset=='SSV2':
                nn.init.zeros_(l.weight)
                nn.init.zeros_(l.bias)
            return l
        
        def make_up():
            l = nn.Conv3d(config.bottle_dim, dim, (1,1,1), (1,1,1), (0,0,0))
            if config.dataset =='HMDB51' or self.config.dataset=='UCF_101' or self.config.dataset=='SSV2':
                nn.init.zeros_(l.weight)
                nn.init.zeros_(l.bias)
            return l
        
        
        if self.config.tp_adapter_mlp:
            
            if self.config.adapter_str == 'sequential':
             
                self.dpconv = [nn.Conv3d(config.bottle_dim, config.bottle_dim, (3,1,1), padding=(1,0,0), groups=config.bottle_dim)]
                sub_0 = [make_down()] + self.dpconv + [make_up()]
                self.net_l = nn.Sequential(*sub_0)
            
            elif self.config.adapter_str =="gumbel" :
                
                
              
                self.inter_out   = None
                self.inter_key   = []
              

                filter_none = lambda x:[i for i in x if i is not None]
                def get_value(obj, key, dft):
                    val = getattr(obj, key, dft)
                    val = val if val is not None else dft
                    return val

                # conv group
                groups = get_value(config, "conv_group", config.bottle_dim)
                # instance norm
                use_in = get_value(config, "adapter_in", False)
                # branch names
                br_key = get_value(config, "branch_key", "sp_tp_relu")

                sub_0 = [
                    None,
                    nn.Conv3d(config.bottle_dim, config.bottle_dim, (1, 3, 3), padding=(0,1,1), groups=groups),
                    # nn.GELU(),
                    None,
                ]
                sub_1 = [
                    None,
                    nn.Conv3d(config.bottle_dim, config.bottle_dim, (3, 1, 1), padding=(1,0,0), groups=groups),
                    # nn.GELU(),
                    None,
                ]
                sub_2 = [
                    None,
                    nn.ReLU(),
                    None,
                ]

                if use_in:
                    # sub_0[0] = nn.InstanceNorm3d(config.bottle_dim)
                    # sub_1[0] = nn.InstanceNorm3d(config.bottle_dim)
                    sub_0[0] = nn.BatchNorm3d(config.bottle_dim)
                    sub_1[0] = nn.BatchNorm3d(config.bottle_dim)
                    sub_2[0] = nn.BatchNorm3d(config.bottle_dim)
                
                sub_0 = filter_none(sub_0)
                sub_1 = filter_none(sub_1)
                sub_2 = filter_none(sub_2)

                
                sub_0 = [make_down()] + sub_0 + [make_up()]
                sub_1 = [make_down()] + sub_1 + [make_up()]
                sub_2 =[make_down()] + sub_2 + [make_up()]

                pre, post = nn.Identity(), nn.Identity()
               
                net_l = [nn.Sequential(*sub_0), nn.Sequential(*sub_1), nn.Sequential(*sub_2)]
                names = ["sp", "tp", "relu" ]

                index = [names.index(k) for k in br_key.split("_")]
                net_l = [ net_l[i] for i in index ]
                names = [ names[i] for i in index ]
                logit = [ 1.0 for i in index]

                self.gumbel = GumbelNetwork(
                    pre, net_l, post,
                    paramnetwork(torch.as_tensor(logit)),
                    self.config.tau_init
                )
                self.inter_key = names
        
        if self.config.tp_adapter_att:
            
            if self.config.adapter_str == 'sequential':
             
                self.dpconv_a = [nn.Conv3d(config.bottle_dim, config.bottle_dim, (3,1,1), padding=(1,0,0), groups=config.bottle_dim)]
                sub_0_a = [make_down()] + self.dpconv + [make_up()]
                self.net_l_a = nn.Sequential(*sub_0)
            
            elif self.config.adapter_str == "gumbel":
                
                
               
                self.inter_out_a   = None
                self.inter_key_a   = []
         

                filter_none_a = lambda x:[i for i in x if i is not None]
                def get_value(obj, key, dft):
                    val = getattr(obj, key, dft)
                    val = val if val is not None else dft
                    return val

                # conv group
                groups = get_value(config, "conv_group", config.bottle_dim)
                # instance norm
                use_in = get_value(config, "adapter_in", False)
                # branch names
                br_key = get_value(config, "branch_key", "sp_tp_relu")

                sub_0_a = [
                    None,
                    nn.Conv3d(config.bottle_dim, config.bottle_dim, (1, 3, 3), padding=(0,1,1), groups=groups),
                    # nn.GELU(),
                    None,
                ]
                sub_1_a = [
                    None,
                    nn.Conv3d(config.bottle_dim, config.bottle_dim, (3, 1, 1), padding=(1,0,0), groups=groups),
                    # nn.GELU(),
                    None,
                ]
                sub_2_a = [
                    None,
                    nn.ReLU(),
                    None,
                ]

                if use_in:
                    # sub_0[0] = nn.InstanceNorm3d(config.bottle_dim)
                    # sub_1[0] = nn.InstanceNorm3d(config.bottle_dim)
                    sub_0_a[0] = nn.BatchNorm3d(config.bottle_dim)
                    sub_1_a[0] = nn.BatchNorm3d(config.bottle_dim)
                    sub_2_a[0] = nn.BatchNorm3d(config.bottle_dim)
                
                sub_0_a = filter_none(sub_0_a)
                sub_1_a = filter_none(sub_1_a)
                sub_2_a = filter_none(sub_2_a)

         
                sub_0_a = [make_down()] + sub_0_a + [make_up()]
                sub_1_a = [make_down()] + sub_1_a + [make_up()]
                sub_2_a =[make_down()] + sub_2_a + [make_up()]

                pre_a, post_a = nn.Identity(), nn.Identity()
                

                net_l_a = [nn.Sequential(*sub_0_a), nn.Sequential(*sub_1_a), nn.Sequential(*sub_2_a)]
                names = ["sp", "tp", "relu" ]

                index = [names.index(k) for k in br_key.split("_")]
                net_l_a = [ net_l_a[i] for i in index ]
                names = [ names[i] for i in index ]
                logit = [ 1.0 for i in index]

                self.gumbel_a = GumbelNetwork(
                    pre_a, net_l_a, post_a,
                    paramnetwork(torch.as_tensor(logit)),
                    self.config.tau_init
                )
                self.inter_key_a = names

    def forward(self, x):
        if self.gamma_1 is None:
            B,h,d = x.shape
            t = self.config.frame
            H = int(math.sqrt(h/(self.config.frame)))
           
            if self.config.tp_adapter_att:                
                x_bcdhw = x.transpose(1,2).reshape(B,d,t,H,H)
                
                if self.config.adapter_str_a == 'sequential':       
                    x_adapt =  self.net_l_a(x_bcdhw) 
                
                elif self.config.adapter_str_a =='gumbel':
                    x_adapt, aux = self.gumbel_a(x_bcdhw)

                    self.inter_out_a = aux["log_p"]
                
                tp_adapt = x_adapt.flatten(2).transpose(1,2)
                x = x + self.config.scale*tp_adapt
            
            x = x + self.drop_path(self.attn(self.norm1(x)))
            
           
            if self.config.tp_adapter_mlp:                
                x_bcdhw = x.transpose(1,2).reshape(B,d,t,H,H)
                
                if self.config.adapter_str == 'sequential':       
                    x_adapt =  self.net_l(x_bcdhw) 
                
                elif self.config.adapter_str == 'gumbel':
                    x_adapt, aux = self.gumbel(x_bcdhw)

                    self.inter_out = aux["log_p"]
                
                tp_adapt = x_adapt.flatten(2).transpose(1,2)
            
            residual = x
            
            x =  self.drop_path(self.mlp(self.norm2(x)))         
        
            if self.config.tp_adapter_mlp:           
                x = x + self.config.scale*tp_adapt
            
            x = residual + x
           
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]), 
                            stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid, cur_frame=-1, pre_n_position=1568,trainable=False): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    if n_position // cur_frame * 8 != pre_n_position and cur_frame != -1:
        T = 8 # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
        print(f'Interpolate the position embedding')
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
        sinusoid_table = torch.nn.functional.interpolate(
            sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    if cur_frame != -1 and cur_frame != 8:
        print(f'Pretraining uses 8 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = 8 # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    if n_position == pre_n_position:
        return sinusoid_table
    else:
        if trainable:
            print("Use learnable position embedding")
            return nn.Parameter(sinusoid_table, requires_grad=True)
        else:
            print("Use sin position embedding")
            # return nn.Parameter(sinusoid_table, requires_grad=False)
            return sinusoid_table


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 fc_drop_rate=0., 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=1,
                 use_checkpoint=False,
                 checkpoint_num=0,
                 use_mean_pooling=True,
                 config = None,
                ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
       
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')
        
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            if patch_size == 14:
                pre_n_position = 2048
            else:
                pre_n_position = 1568
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim, all_frames // tubelet_size,
                pre_n_position=pre_n_position,trainable = config.trainable
            )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values,config=config)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.out=0
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
      
        x = self.patch_embed(x)
        B, _, _ = x.size()
        
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint and idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        x = self.norm(x)
        
        if self.fc_norm is not None:
            self.out=self.fc_norm(x.mean(1))
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        
        x = self.head(self.fc_dropout(x))
        return x


@register_model
def gb_vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def gb_vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def gb_vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def gb_vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    import time
    import numpy as np
    from easydict import EasyDict
    tuning_config = EasyDict(        
        tp_adapter_att = True,
        tp_adapter_mlp = True,
        bottle_dim= 64,
        conv_group= None,
        adapter_in= True,
        branch_key= "sp_tp_relu",
        dataset = "Kinetics_sparse",
        adapter_str = "gumbel",
        adapter_str_a = "gumbel",
        frame = 16,
        scale= 1.0,
        trainable = False,
        tau_init = 1.0
    )
    model = gb_vit_base_patch16_224(config = tuning_config)
    print(model(torch.rand(1, 3, tuning_config.frame, 224, 224)).shape)
  