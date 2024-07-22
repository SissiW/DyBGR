import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from apex import amp
from apex.parallel import DistributedDataParallel
import pprint
import time
import torch.nn as nn
import os
from tqdm import trange
# import wandb
import logging
import multiprocessing
from functools import partial
import numpy as np
import PIL
# from tap import Tap
from typing_extensions import Literal
import os.path as osp
import argparse
from thop import profile

from sampler import UniqueClassSempler
from proxy_anchor.dataset import CUBirds, SOP, Cars
from proxy_anchor.dataset.Inshop import Inshop_Dataset
from hyptorch.pmath import dist_matrix
from net.model import init_model
from hyptorch.helpers import get_emb, evaluate, ensure_path

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def contrastive_loss(x0, x1, tau, hyp_c):
    # x0 and x1 - positive pair 
    # tau - temperature 
    # hyp_c - hyperbolic curvature, "0" enables sphere mode
    if hyp_c == 0:
        dist_f = lambda x, y: x @ y.t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    logits00 = dist_f(x0, x0) / tau - eye_mask
    logits01 = dist_f(x0, x1) / tau
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    stats = {
        "logits/min": logits01.min().item(),
        "logits/mean": logits01.mean().item(),
        "logits/max": logits01.max().item(),
        "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
    }
    return loss, stats

if __name__ == "__main__":
    data_dir = '/DATA/gblav1/wxx/data/metric_learning/'
    parser = argparse.ArgumentParser(description="Training")
    # about dataset and network
    parser.add_argument('-model', type=str, default="vit_small_patch16_224", \
           help="model name from timm or torch.hub, i.e. vit_small_patch16_224_in21k, deit_small_distilled_patch16_224, vit_small_patch16_224, dino_vits16")
    parser.add_argument('-ds', type=str, default='SOP', choices=['SOP', 'CUB', 'Cars', 'Inshop'])
    parser.add_argument('-path', type=str, default=data_dir)
    parser.add_argument('-num_samples', type=int, default=9, help='how many samples per each category in batch')
    parser.add_argument('-bs', type=int, default=459, \
            help='batch size per GPU, e.g. --num_samples 3 --bs 900 means each iteration we sample 300 categories with 3 samples')
    parser.add_argument('-emb', type=int, default=384, help='output embedding size')
    parser.add_argument('-freeze', type=int, default=0, \
                                    help='number of blocks in transformer to freeze, None - freeze nothing, 0 - freeze only patch_embed')
    parser.add_argument('-clip_r', type=float, default=2.3, help='feature clipping radius')
    parser.add_argument('-resize', type=int, default=224, help='image resize')
    parser.add_argument('-crop', type=int, default=224, help='center crop after resize')
    parser.add_argument('-save_emb', type=bool, default=True, help='save embeddings of the dataset after training ')
    # training
    parser.add_argument('-lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('-lrt', type=float, default=3e-5, help='learning rate')
    parser.add_argument('-t', type=float, default=0.2, help='cross-entropy temperature')
    parser.add_argument('-max_epoch', type=int, default=100, help='number of epochs 100')
    parser.add_argument('-hyp_c', type=float, default=0.1, help='hyperbolic c, "0" enables sphere mode')
    parser.add_argument('-eval_ep', type=str, default='[100]', \
                  help='epochs for evaluation, [] or range "r(start,end,step)", e.g. "r(10,70,20)+[200]" means 10, 30, 50, 200')
    parser.add_argument('-emb_name', type=str, default='emb', help='filename for embeddings')
    parser.add_argument('-dist_training', type=bool, default=False, help='If train with multi-gpu ddp mode, options: True, False')
    parser.add_argument('-local_rank', default=0, type=int, help='set automatically for distributed training')
    parser.add_argument('-savepath', type=str, default='./logs/GraphLearning/backbone')
    parser.add_argument("-num_classes", type=int, default=98, help="number of centers for each categories")
    # loss
    parser.add_argument('-alpha', type=float, default=0.01, help='the balaced hyper-parameter for loss')
    # graph learning
    parser.add_argument('-depth', type=int, default=1, help='the number of Former layers')
    parser.add_argument('-num_heads', type=int, default=8, help='the number of heads in Former')
    parser.add_argument('-K', type=int, default=100, help='k-nearast neighborhood')
    parser.add_argument('-T', type=int, default=1, help='the iteration time')
    parser.add_argument('-epsilon', type=float, default=0.5, help='the balanced hyper-parameter between the learned S and the identity matrix')
    parser.add_argument('-lam', type=float, default=0.3, help='the weight of FF^T')
    parser.add_argument('-beta', type=float, default=0.5, help='the weight between D and L')
    # resum
    parser.add_argument('-resume',action='store_true',help='Whether the breakpoint is resumed')
    parser.add_argument('-resum_path', type=str, default='./logs',help='dir of breakpoint')

    cfg = parser.parse_args()
    pprint(vars(cfg))
    
    # cfg.savepath = osp.join(cfg.savepath, 'SGFormer/'+str(cfg.emb)+'/'+cfg.ds+'/'+'l'+str(cfg.lam)+'/'+ cfg.model.split('_')[0])
    cfg.savepath = osp.join(cfg.savepath, cfg.ds)
    ensure_path(cfg.savepath)

    if cfg.dist_training:
        torch.cuda.set_device(cfg.local_rank)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print('world_size: ', world_size)
    if world_size > 1:
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_size = torch.distributed.get_world_size()
        print('1', world_size)

    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tr = T.Compose(
        [
            T.RandomResizedCrop(cfg.crop, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),           
            T.Normalize(*mean_std),
        ]
    )
    # T.RandomErasing(0.3),
    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    ds_class = ds_list[cfg.ds]
    ds_train = ds_class(cfg.path, "train", train_tr)
    assert len(ds_train.ys) * cfg.num_samples >= cfg.bs * world_size
    # print('01', len(ds_train.ys), ds_train.ys)  # [0-99]
    # print('02', len(ds_train.I), ds_train.I)
    #image_paths: [...,'/DATA/gblav1/wxx/data/metric_learning/CUB_200_2011/images/100.Brown_Pelican/Brown_Pelican_0141_94533.jpg']
    sampler = UniqueClassSempler(
        ds_train.ys, cfg.num_samples, cfg.local_rank, world_size
    )
    dl_train = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=cfg.bs,
        num_workers= 8,
        pin_memory=True,
        drop_last=False,  # CUB,Cars is False
    )
    # multiprocessing.cpu_count() // world_size,

    model = init_model(cfg)
    optimizer = optim.AdamW([{'params': model.body.parameters(), 'lr': cfg.lr},
                             {'params': model.head.parameters(), 'lr': cfg.lr},
                             {'params': model.blocks.parameters(), 'lr': cfg.lrt}], lr=cfg.lr, weight_decay=5e-5)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    

    if world_size > 1:
        model = DistributedDataParallel(model, delay_allreduce=True)

    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(cfg.savepath, name + '.pth'))

    loss_f = partial(contrastive_loss, tau=cfg.t, hyp_c=cfg.hyp_c)
    get_emb_f = partial(
        get_emb,
        model=model,
        ds=ds_class,
        path=cfg.path,
        mean_std=mean_std,
        world_size=world_size,
        resize=cfg.resize,
        crop=cfg.crop,
    )
    eval_ep = eval(cfg.eval_ep.replace("r", "list(range").replace(")", "))"))

    cudnn.benchmark = True
    time_sum = 0
    time_start = time.time()
    max_rh = 0
    max_rb = 0
    max_rh_all = []
    max_rb_all = []
    max_rh_ep = 0
    max_rb_ep = 0
    torch.cuda.empty_cache()
    print('Start time {:.5f}'.format(time_start))
    for ep in range(cfg.max_epoch):
        stats_ep = []
        for i, (x, y) in enumerate(dl_train):  
            label = y
            # i = the total number of classes N*9/bs
            y = y.view(len(y) // cfg.num_samples, cfg.num_samples)  
            
            assert (y[:, 0] == y[:, -1]).all()
            s = y[:, 0].tolist()  
            
            assert len(set(s)) == len(s)

            x = x.cuda(non_blocking=True)  

            z = model(x, label).view(len(x)*2 // cfg.num_samples, cfg.num_samples, cfg.emb)  
            z1, z2 = z[:len(y)], z[len(y):]

            torch.cuda.empty_cache()
            loss = 0
            loss1 = 0
            loss2 = 0
            for i in range(cfg.num_samples):
                for j in range(cfg.num_samples):
                    if i != j:
                        l, s = loss_f(z1[:, i], z1[:, j])
                        loss1 += l
                        l2, s2 = loss_f(z2[:, i], z2[:, j])
                        loss2 += l2
                        loss = cfg.alpha * loss1 + (1 - cfg.alpha) * loss2

                        stats_ep.append({**s, "loss": l.item()})
                torch.cuda.empty_cache()

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 3)
            optimizer.step()

        if ep % 5 == 0:
            rh, rb, rh_all, rb_all = evaluate(get_emb_f, cfg.ds, cfg.hyp_c)
            if max_rh < rh:
                max_rh = rh
                max_rh_all = rh_all
                max_rh_ep = ep
            if max_rb <rb:
                max_rb = rb
                max_rb_all = rb_all
                max_rb_ep = ep
                if cfg.save_emb:
                    ds_type = "gallery" if cfg.ds == "Inshop" else "eval"
                    x, y = get_emb_f(ds_type=ds_type)
                    x, y = x.float().cpu(), y.long().cpu()
                    torch.save((x, y), cfg.savepath +'/'+ cfg.emb_name + "_eval.pt")

                    x, y = get_emb_f(ds_type="train")
                    x, y = x.float().cpu(), y.long().cpu()
                    torch.save((x, y), cfg.savepath +'/' + cfg.emb_name + "_train.pt")

                    save_model('max_acc')
        torch.cuda.empty_cache()

        if cfg.local_rank == 0:
            stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
            if (ep + 1) in eval_ep:
                stats_ep = {"recall": rh, "recall_b": rb, **stats_ep}
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'epoch[', ep, '/', cfg.max_epoch, \
                    '], logits/mean:', stats_ep['logits/mean'], ', logits/acc:', stats_ep['logits/acc'], \
                    ', loss:', stats_ep['loss'], ', lr:', optimizer.param_groups[0]['lr'])
        # lr_scheduler.step()

    time_end = time.time()
    time_sum = time_end- time_start
    print('End time {:.5f}'.format(time_end))
    print('Running time {:.5f}'.format(time_sum))
    if cfg.ds == "CUB" or cfg.ds == "Cars":
        print('Epoch:', max_rh_ep, 'gets the best results! \n Recall@[1, 2, 4, 8, 16, 32]=', max_rh_all, 'where hyperbolic c=', cfg.hyp_c)
        print('Epoch:', max_rb_ep, 'gets the best results! \n Recall@[1, 2, 4, 8, 16, 32]=', max_rb_all)
    elif cfg.ds == "SOP":
        print('Epoch:', max_rh_ep, 'gets the best results! \n Recall@[1, 10, 100, 1000]=', max_rh_all, 'where hyperbolic c=', cfg.hyp_c)
        print('Epoch:', max_rb_ep, 'gets the best results! \n Recall@[1, 10, 100, 1000]=', max_rb_all)
    else:
        print('Epoch:', max_rh_ep, 'gets the best results! \n Recall@[1, 10, 20, 30, 40, 50]=', max_rh_all, 'where hyperbolic c=', cfg.hyp_c)
        print('Epoch:', max_rb_ep, 'gets the best results! \n Recall@[1, 10, 20, 30, 40, 50]=', max_rb_all)

    
