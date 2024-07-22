# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)  # torch.Size([882, 384])
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss_simple(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

class SoftTripleLoss(nn.Module):

    """Qi Qian, et al.,
    `SoftTriple Loss: Deep Metric Learning Without Triplet Sampling`,
    https://arxiv.org/abs/1909.05235
    """

    def __init__(
        self,
        embedding_dim: int,
        num_categories: int,
        use_regularizer: bool = True,
        num_initial_center: int = 2,
        similarity_margin: float = 0.1,
        coef_regularizer1: float = 1e-2,
        coef_regularizer2: float = 1e-2,
        coef_scale_softmax: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """Constructor
        Args:
            embedding_dim: dimension of inputs to this module (N x embedding_dim)
            num_categories: total category count to classify
            num_initial_center: initial number of centers for each categories
            similarity_margin: margin term as is in triplet loss
            coef_regularizer1: entropy regularizer for dictibution over classes
            coef_regularizer2: regularizer for cluster variancce.
            coef_scale_softmax: scaling factor before final softmax op
            device: device on which this loss is computed
        """

        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_categories = num_categories
        self.use_regularizer = use_regularizer
        self.num_initial_center = num_initial_center
        self.delta = similarity_margin
        self.gamma_inv = 1 / coef_regularizer1
        self.tau = coef_regularizer2
        self.lambda_ = coef_scale_softmax
        self.device = device
        self.fc_hidden = nn.Linear(
            embedding_dim, num_categories * num_initial_center
        ).to(device)
        nn.init.xavier_normal_(self.fc_hidden.weight)
        self.base_loss = nn.CrossEntropyLoss().to(device)
        self.softmax = nn.Softmax(dim=2).to(device)

    def infer(self, embedding):
        weight = F.normalize(self.fc_hidden.weight)
        x = F.linear(embedding, weight).view(
            -1, self.num_categories, self.num_initial_center
        )
        x = self.softmax(x.mul(self.gamma_inv)).mul(x).sum(dim=2)
        return x

    def cluster_variance_loss(self):
        weight = F.normalize(self.fc_hidden.weight)
        loss = 0.0
        for i in range(self.num_categories):
            weight_sub = weight[
                i * self.num_initial_center : (i + 1) * self.num_initial_center
            ]
            subtraction_norm = 1.0 - torch.matmul(
                weight_sub, weight_sub.transpose(1, 0)
            )
            subtraction_norm[subtraction_norm <= 0.0] = 1e-10
            loss += torch.sqrt(2 * subtraction_norm.triu(diagonal=1)).sum()

        loss /= (
            self.num_categories
            * self.num_initial_center
            * (self.num_initial_center - 1)
        )
        return loss

    def forward(self, embeddings, labels):
        h = self.infer(embeddings)
        one_hot = torch.zeros(h.size(), device=self.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        h = h - self.delta * one_hot
        h.mul_(self.lambda_)
        clf_loss = self.base_loss(h, labels)
        if not self.use_regularizer:
            return clf_loss

        var_loss = self.cluster_variance_loss()
        return clf_loss + self.tau * var_loss