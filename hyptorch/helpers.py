import torch
import os
import torchvision.transforms as T
from torch.utils.data import DataLoader
from proxy_anchor.utils import calc_recall_at_k
from hyptorch.pmath import dist_matrix
import PIL
import multiprocessing
import os.path as osp
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))


def evaluate(get_emb_f, ds_name, hyp_c):
    if ds_name != "Inshop":
        emb_head = get_emb_f(ds_type="eval")  # cub dataset: torch.Size([5924, 128]) torch.Size([5924])
        emb_body = get_emb_f(ds_type="eval", skip_head=True)
        recall_head, recall_head_all = get_recall(*emb_head, ds_name, hyp_c)
        recall_body, recall_body_all = get_recall(*emb_body, ds_name, 0)
    else:
        emb_head_query = get_emb_f(ds_type="query")
        emb_head_gal = get_emb_f(ds_type="gallery")
        emb_body_query = get_emb_f(ds_type="query", skip_head=True)
        emb_body_gal = get_emb_f(ds_type="gallery", skip_head=True)
        recall_head, recall_head_all = get_recall_inshop(*emb_head_query, *emb_head_gal, hyp_c)
        recall_body, recall_body_all = get_recall_inshop(*emb_body_query, *emb_body_gal, 0)
    return recall_head, recall_body, recall_head_all, recall_body_all


def get_recall(x, y, ds_name, hyp_c):
    '''
    cub dataset:
                x: torch.Size([5924, 128]) 
                y: torch.Size([5924]) 
    ds_name: CUB
    '''
    # print('0', x.size(), y.size(), y)  # torch.Size([5924, 384]) torch.Size([5924])
    if ds_name == "CUB" or ds_name == "Cars":
        k_list = [1, 2, 4, 8, 16, 32]
    elif ds_name == "SOP":
        k_list = [1, 10, 100, 1000]

    if hyp_c > 0:
        dist_m = torch.empty(len(x), len(x), device="cuda")
        for i in range(len(x)):
            dist_m[i : i + 1] = -dist_matrix(x[i : i + 1], x, hyp_c)
    else:
        dist_m = x @ x.T

    y_cur = y[dist_m.topk(1 + max(k_list), largest=True)[1][:, 1:]]  # torch.Size([5924, 32])
    
    # =================== new metric =======================
    gt = y.cpu().numpy()
    pred = y_cur[:,:1].reshape(-1).cpu().numpy()  # torch.Size([5924])
    # print('01', gt.shape, pred.shape)
    # 计算NMI评价指标
    nmi = normalized_mutual_info_score(gt, pred)
    print("NMI score:", nmi)
    # 计算RI评价指标
    ri = adjusted_rand_score(gt, pred)
    print("RI score:", ri)
    # print('1', y_cur.size(), y_cur)
    # print(zz)
    # =====================================================
    
    y = y.cpu()
    y_cur = y_cur.float().cpu()
    recall = [calc_recall_at_k(y, y_cur, k) for k in k_list]
    print('Recall@K', recall[0], recall[1], recall[2], recall[3])
    return recall[0], recall


def get_recall_inshop(xq, yq, xg, yg, hyp_c):
    if hyp_c > 0:
        dist_m = torch.empty(len(xq), len(xg), device="cuda")
        for i in range(len(xq)):
            dist_m[i : i + 1] = -dist_matrix(xq[i : i + 1], xg, hyp_c)
    else:
        dist_m = xq @ xg.T

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0
        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]
            thresh = torch.max(pos_sim).item()
            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
        return match_counter / m

    recall = [recall_k(dist_m, yq, yg, k) for k in [1, 10, 20, 30, 40, 50]]
    print('Recall@K', recall[0], recall[1], recall[2], recall[3])
    return recall[0], recall


def get_emb(
    model,
    ds,
    path,
    mean_std,
    resize=224,
    crop=224,
    ds_type="eval",
    world_size=1,
    skip_head=False,
):
    eval_tr = T.Compose(
        [
            T.Resize(resize, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )
    ds_eval = ds(path, ds_type, eval_tr)
    if world_size == 1:
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(ds_eval)
    dl_eval = DataLoader(
        dataset=ds_eval,
        batch_size=100,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )
    model.eval()
    x, y = eval_dataset(model, dl_eval, skip_head)  # the eval of cub dataset: torch.Size([5924, 128]) torch.Size([5924])
    y = y.cuda()
    if world_size > 1:
        print('*********************')
        all_x = [torch.zeros_like(x) for _ in range(world_size)]
        all_y = [torch.zeros_like(y) for _ in range(world_size)]
        torch.distributed.all_gather(all_x, x)
        torch.distributed.all_gather(all_y, y)
        x, y = torch.cat(all_x), torch.cat(all_y)
    model.train()
    return x, y


def eval_dataset(model, dl, skip_head):
    all_x, all_y = [], []
    for x, y in dl:  # torch.Size([100, 3, 224, 224]) label y: torch.Size([100]) -> 100*i = all sample numbers of (100-200) classes in cub dataset
        with torch.no_grad():
            x = x.cuda(non_blocking=True)
            logits = model(x, y, skip_head=skip_head)
            all_x.append(logits)  # the outputs of model: torch.Size([100, 128])
        all_y.append(y)
    return torch.cat(all_x), torch.cat(all_y)

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print ('create folder:',path)
        os.makedirs(path)