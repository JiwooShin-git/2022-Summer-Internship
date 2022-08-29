import copy
from time import time
import torch
import numpy as np
import logging
import torch.nn.functional as F
import torch.nn as nn
from visualization import save_image
from cka import linear_CKA
from ptflops import get_model_complexity_info

def gate_function(x):
    temp = 5
    return 1/(1 + torch.exp(-1*temp*x))

def check_params(model, img_size):
    model_check = copy.deepcopy(model)
    flops, params = get_model_complexity_info(model_check, (3, img_size, img_size), False, False)
    del model_check
    return flops, params


def create_logging(path_log):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path_log)
    logger.addHandler(file_handler)
    return logger


def count_classes(data):
    if data == 'setting':
        return 100, 500
    elif data == 'TINY':
        return 200, 500
    elif data == 'IMAGENET':
        return 1000, 1300
    else:
        raise KeyError('Data is not defined')


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

    return losses.avg, top1.avg, top5.avg

def train_our(model, optimizer, criterion, train_loader, device, gamma):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        for name, param in model.named_parameters():
            if 'gate' in name:
                loss += gamma * torch.sum(param)
        loss.backward()

        optimizer.step()
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

    return losses.avg, top1.avg, top5.avg


def train_kl(module_list, optimizer, criterion, train_loader, device, args, epoch, depth_weight_list):
    for module in module_list:
        module.train()
    # module_list[-1] -> model_t
    module_list[-1].eval()

    model_s = module_list[0]
    model_t = module_list[-1]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion_ce, criterion_kl, criterion_kd = criterion

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            feat_t, output_t = model_t(inputs, is_feat=True)
            feat_t = [f.detach() for f in feat_t]
        feat_s, output_s = model_s(inputs, is_feat=True)

        loss_ce = criterion_ce(output_s, targets)
        loss_kl = criterion_kl(output_s, output_t)

        if args.distill == 'fit':
            f_s = module_list[1](feat_s[-3])
            f_t = feat_t[-3]
            loss_kd = criterion_kd(f_s, f_t)
        elif args.distill == 'att':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        else:
            loss_kd = 0

        # save CKA
        if epoch == 240:
            if batch_idx == 0:
                cka_results = np.empty([len(feat_t)-1, len(feat_s)-1])
                avg_pool_2d = nn.AdaptiveAvgPool2d((1,1))
                
            for i in range(len(feat_t)-1):
                for j in range(len(feat_s)-1):
                    feat_t_pool = avg_pool_2d(feat_t[i]).squeeze()
                    feat_s_pool = avg_pool_2d(feat_s[j]).squeeze()
                    cka_results[i][j] += linear_CKA(feat_t_pool.cpu().numpy(), feat_s_pool.detach().cpu().numpy())

        loss = args.gamma * loss_ce + args.alpha * loss_kl + args.beta * loss_kd

        depth_weight = 1
        gate_sum = 0

        for name, param in module_list.named_parameters():
            
            if 'gate' in name:
                if args.use_depth:
                    if 'layer1' in name:
                        depth_weight = depth_weight_list[0]
                    elif 'layer2' in name:
                        depth_weight = depth_weight_list[1]
                    elif 'layer3' in name:
                        depth_weight = depth_weight_list[2]
                    if len(depth_weight_list) == 4 and 'layer4' in name:
                        depth_weight = depth_weight_list[3]
                        
                if args.optimizer == 'sss':
                    loss += depth_weight * args.lambda1 * torch.sum(torch.abs(param))
                else:
                    # triangle
                    # loss += depth_weight * args.lambda1 * torch.sum(torch.pow(0.5 - torch.abs(param-0.5), 1))
                    # rounded
                    if epoch > 30:
                        gate_sum += gate_function(param)
                        loss += depth_weight * args.lambda1 * 2 * torch.sum(0.25 - torch.pow((gate_function(param)-0.5), 2))
                    # loss += depth_weight * args.lambda1 * torch.sum(0.5 - torch.abs(gate_function(param)-0.5))       

        if args.optimizer != 'sss':
            if epoch > 30:
                loss += args.lambda2 * torch.pow((torch.tensor(args.remain_block_num).to(loss.device) - gate_sum.squeeze()), 2)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)

        optimizer.step()
        acc1, acc5 = accuracy(output_s, targets, topk=(1, 5))

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)
    
    # if epoch == 240:
    #     cka_results = cka_results / len(train_loader)
    #     save_image(cka_results, f'{args.distill}_{args.model_t}_{args.model_s}_{epoch}.png')
        
    return losses.avg, top1.avg, top5.avg #, np.mean(cls), np.mean(ils)


def test(model, test_loader, device):
    top1 = AverageMeter()
    top5 = AverageMeter()
    tot_infer_time = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            s = time()
            outputs = model(inputs)
            # inter_time: elapsed time for one batch
            infer_time = time() - s
            tot_infer_time += infer_time

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            batch_size = targets.size(0)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
        avg_batch_infer_time = tot_infer_time / len(test_loader)

    return top1.avg, top5.avg, avg_batch_infer_time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr


class LossKD:
    def __init__(self, temperature):
        self.T = temperature
        self.KLD = nn.KLDivLoss(reduction='batchmean') # TODO: reduction mean? batchmean??

    def __call__(self, outputs, labels, teacher_outputs):
        kld_loss = self.KLD(F.log_softmax(outputs/self.T, dim=1), F.softmax(teacher_outputs/self.T, dim=1)) * self.T ** 2
        return kld_loss


