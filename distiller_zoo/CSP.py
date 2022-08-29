import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoryRelation(nn.Module):
    def __init__(self, args):
        super(CategoryRelation, self).__init__()
        self.wc = args.wc
        self.wi = args.wi
        self.wp = args.wp
        self.category_loss = CategoricalSimilarity()
        self.instance_loss = InstanceSimilarity(args)
        self.instance_pair_loss = InstancePairSimilarity()

    def forward(self, c_s, c_t, f_s, f_t):
        loss = 0
        cl, il, pl = 0, 0, 0
        if self.wc > 0:
            cl = self.category_loss(c_s, c_t, f_s)
            loss += self.wc * cl
        if self.wi > 0:
            il = self.instance_loss(c_s, c_t, f_s, f_t)
            loss += self.wi * il
        if self.wp > 0:
            pl = self.instance_pair_loss(f_s, f_t)
            loss += self.wp * pl
        return loss, self.wc * cl, self.wi * il, self.wp * pl


class InstancePairSimilarity(nn.Module):
    def __init__(self):
        super(InstancePairSimilarity, self).__init__()
        self.distance = CategoricalSimilarity.distance_sp

    def forward(self, f_s, f_t):
        with torch.no_grad():
            diff_t = self.distance(f_t)
        diff_s = self.distance(f_s)
        diff = (diff_t - diff_s)
        loss = (diff * diff).mean()
        return loss


class InstanceSimilarity(nn.Module):
    def __init__(self, args):
        super(InstanceSimilarity, self).__init__()
        self.args = args
        if self.args.f_dim_t != self.args.f_dim_s:
            self.layer_t = nn.Linear(args.f_dim_t, args.feat_dim)
            self.layer_s = nn.Linear(args.f_dim_s, args.feat_dim)

    def forward(self, c_s, c_t, f_s, f_t):
        if self.args.f_dim_t != self.args.f_dim_s:
            diff_t = self.layer_t(f_t - c_t)
            diff_s = self.layer_s(f_s - c_s)
        else:
            diff_t = f_t - c_t
            diff_s = f_s - c_s
        t_mean, s_mean = diff_t.norm(2, dim=1).mean(), diff_s.norm(2, dim=1).mean()
        loss = ((diff_t / t_mean - diff_s / s_mean) * (diff_t / t_mean - diff_s / s_mean)).sum(1).mean()/2
        return loss


class CategoricalSimilarity(nn.Module):
    def __init__(self):
        super(CategoricalSimilarity, self).__init__()

    def forward(self, c_s, c_t, f_s):
        # TODO: normalization
        bsz = c_s.size(0)
        with torch.no_grad():
            dist_s = self.distance(c_s, is_mean=False)
            dist_s_mean = dist_s[dist_s > 0].mean()
            dist_t = self.distance(c_t)
            delta = (dist_t - torch.div(dist_s, dist_s_mean)) / (2 * dist_s + 0.0000001)

        c2f = torch.mm(f_s, c_s.t())
        loss = - delta * (-c2f + c2f.diag().view(-1, 1))
        loss = loss.sum() / (bsz * (bsz - 1))
        return loss

    @staticmethod
    def distance(c, is_mean=True):
        # Stable but not efficient (memory)
        diff = c.unsqueeze(1) - c.unsqueeze(0)
        dist = torch.sqrt(torch.clamp_min(torch.sum(diff * diff, -1), 0) + 0.000001)
        # normalization
        if is_mean:
            mean = dist[dist > 0].mean()
            dist = torch.div(dist, mean)
            # dist = torch.div(dist - dist.mean(), dist.std())
        return dist

    @staticmethod
    def distance_sp(c):
        dist = torch.mm(c, c.t())
        return torch.nn.functional.normalize(dist)
