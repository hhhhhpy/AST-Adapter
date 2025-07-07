from easydict import EasyDict
import argparse
from datasets.build import build_dataset
import torch.nn as nn
import torch
import utils
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
from timm.models import create_model

import model
from copy import deepcopy

def get_args_parser():
    parser = argparse.ArgumentParser('Spatiotemporal information ratio metric', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--batch_size', default=1, type=int,)       
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)

    # Dataset parameters
    parser.add_argument('--data_path', default=None, type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
   
    parser.add_argument('--prefix', default='', type=str, help='prefix for data')
   
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--use_checkpoint', action='store_true', default=False)
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # custom parameters
    
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='No drop path for linear probe')
    parser.add_argument('--use_mean_pooling', default=True)
    parser.add_argument('--init_scale', default=0.001, type=float)

    # video data parameters
    parser.add_argument('--data_set', default='SSV2',
                        choices=['SSV2', 'HMDB51', 'UCF101','kinetics400'],
                        type=str, help='dataset')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--sampling_rate', type=int, default=2)
    parser.add_argument('--num_sample', type=int, default=1,
                        help='Repeated_aug (default: 1)')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=4)
    parser.add_argument('--test_num_crop', type=int, default=3)
    parser.add_argument('--input_size', default=224, type=int, help='videos input size')
    parser.add_argument('--split', default=' ', type=str, help='split for metadata')
    
    #tp adapter
    parser.add_argument('--tp_adapter_mlp', default=False, action='store_true')
    parser.add_argument('--tp_adapter_att', default=False, action='store_true')
    parser.add_argument('--tp_bottle_dim', default=256, type=int)
    parser.add_argument('--adapter_str', default='sequential', type=str)
    parser.add_argument('--adapter_str_a', default='sequential', type=str)
    parser.add_argument('--scale', default=0.2, type=float)
    parser.add_argument('--tau_init', default=2.0, type=float)
    parser.add_argument('--tp_adapter_in', default=None, type=lambda x:bool(eval(x)))
    parser.add_argument('--tp_branch_key', default=None, type=str)
    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--checkpoint_num', default=50, type=int,
                        help='number of layers for using checkpoint')
    return parser



def load_model(dict,args):
    
    model=create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        use_learnable_pos_emb=False,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        checkpoint_num=args.checkpoint_num,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        config = dict)
    
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    checkpoint = torch.load(args.finetune, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % args.finetune)
    load_uma_dict(model,args)    
    model.to(device)
    
    return model

def load_uma_dict(model,args):
    checkpoint = torch.load(args.finetune,map_location='cpu')
    model_keys = 'model|module'
    checkpoint_model = None
    for model_key in model_keys.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            # print(checkpoint_model)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict
    load_state_dict(model,checkpoint_model)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))    


def new_metric(data):
    avg_new = data/2+0.5
    avg_new = 1-avg_new
    return avg_new

class feature_hook(object):
    from collections import defaultdict
    features= defaultdict(list)
    def __init__(self, name, cpu=False):
        super().__init__()
        self.name = name
        self.cpu  = cpu

    def __call__(self,module,input,output):
        # print(module)
        if self.cpu:                      
            self.features[self.name].append(output.detach().cpu())
        else:           
            self.features[self.name].append(output)


def im2col(image, kernel, padding=0, stride=1):
    col = F.unfold(image, kernel, padding=padding, stride=stride)
    return col

def off_diagonal(square_matrix):
    n,n_ = square_matrix.shape[-2:]
    assert n == n_, f"last 2 dimension should be same, but got {square_matrix.shape}"

    n2_1 = square_matrix.flatten(-2)[..., 1:]
    n2_n = n2_1.unflatten(-1, (n-1, n+1))[..., :, :-1]
    return n2_n.flatten(-2).unflatten(-1, (n, n-1))

if __name__ =="__main__":
    
    #init
    args = get_args_parser()
    args = args.parse_args()   
    utils.init_distributed_mode(args)
    seed = args.seed + utils.get_rank()
   
    device = torch.device(args.device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    #build_dataset
    if args.data_set=='HMDB51':
        args.nb_classes = 51
    
    elif args.data_set == 'SSV2':
        args.nb_classes = 174
    
    elif args.data_set == 'UCF_101':
        args.nb_classes = 101
    
    elif args.data_set == 'kinetics400':
        args.nb_classes = 400
   
    dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    #tuning_config / remove adapter
    tuning_config = EasyDict(
                tp_adapter_mlp=args.tp_adapter_mlp,
                tp_adapter_att=args.tp_adapter_att,
                bottle_dim=args.tp_bottle_dim,              
                dataset = args.data_set,              
                adapter_str = args.adapter_str,  
                adapter_str_a = args.adapter_str_a,             
                frame = args.num_frames,                
                scale=args.scale,             
                adapter_in=args.tp_adapter_in,
                tau_init=args.tau_init,
                trainable = False,             
            )    
    model = load_model(tuning_config,args)
   
   
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    
    layer_dict = {0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1}
    model.patch_embed.register_forward_hook(feature_hook(f'{0}_weight'))
    for i in range(12):
        for j in range(layer_dict[i]):
            model.blocks[i].register_forward_hook(feature_hook(f'{i}_{j}_weight'))
            
    model.norm.register_forward_hook(feature_hook('12_0_weight')) 
    

    sim_dict = {}
    layer_d  = {}      
    layer_l = len(layer_dict)

    for i in range(layer_l):
        for j in range(layer_dict[i]):
            sim_dict[f'{i}_{j}_sp'] = []
            sim_dict[f'{i}_{j}_st'] = []
            layer_d[f'{i}_{j}']     = len(layer_d)       

    #metric calculation   
    with torch.no_grad():          
        for bi, batch in enumerate(metric_logger.log_every(data_loader_val, 1, header)):
            
            data = batch[0] #b,c,d,h,w
            target=batch[1]
            
                
            data = data.to(device, non_blocking=True)
            
            output = model(data)
            layer_l = len(layer_dict)
            for i in range(layer_l):
                for j in range(layer_dict[i]):
                    feat = feature_hook.features[f'{i}_{j}_weight'][0]  # B,T,H,W,C
                    B,D,C = feat.shape
                    
                    feat = feat.reshape(B,-1,14,14,C)
                       
                          
                    B,T,H,W,C = feat.shape
                    
                    
                    def block_cos_sim(blk):
                        
                        blk = F.normalize(blk, dim=1)
                        sim = torch.einsum("bcil,bcjl->bijl", blk, blk)  # B*T, K, K, L                           
                        sim = off_diagonal(sim.permute(0, 3, 1, 2))     # B*T, L, K, K-1                            
                        sim = sim.flatten(-2).mean(-1)  #B*T,L                                         
                        return sim

                    
                    # sp
                    spim = feat.permute(0, 1, 4, 2, 3).flatten(0, 1)
                    col  = im2col(spim, (3, 3), stride=1)     # B*T, C, H, W -> B*T, C*K, L
                    blk  = col.reshape(B*T, C, 9, -1)         # B*T, C, K, L
                    sp_sim = block_cos_sim(blk).mean(dim=-1) # B*T
                    sim_dict[f'{i}_{j}_sp'].append(sp_sim.detach().cpu().reshape(B, T))
                  

                    #st
                    stim = feat.permute(0, 2, 3, 4, 1).flatten(0, 2).unsqueeze(-1) # B*H*W,C,T,1                
                    blk = stim.reshape(B*H*W, C, T, -1)                    
                    st_sim = block_cos_sim(blk).mean(dim=-1)  # B*H*W
                    sim_dict[f'{i}_{j}_st'].append(st_sim.detach().cpu().reshape(B, H, W))
                    
           
                   
            feature_hook.features.clear()
         
        new_me = {}
        for name in ["sp", "st"]:
            x = []
            avg= []
            max_,min_= [], []
            layer_sim = []
            layer_l = len(layer_dict)
            for i in range(layer_l):
                for j in range(layer_dict[i]):
                 
                    sim = torch.cat(sim_dict[f'{i}_{j}_{name}'], dim=0)
                    
                    layer_sim.append(sim.detach().cpu().numpy())
                 
                    sim = sim.flatten()
                    
                    x.append(layer_d[f"{i}_{j}"])
                    avg.append(sim.mean().item())
                    
            avg = np.array(avg)            
            new_me[name] = new_metric(avg)
        
        offset = [1.8511909, 1.9739718, 1.9370402,2.09378, 2.0608604, 2.2353857, 2.3201206, 2.3767254, 2.4827647, 2.5602362, 2.555112, 2.4415195, 2.4415195] #K400 spatiotemporal ratio
        final_metric = [np.log((t*o)/(p)) for p,t,o in zip(new_me["sp"],new_me["st"],offset)]
        print(args.data_set,final_metric)
          
      
