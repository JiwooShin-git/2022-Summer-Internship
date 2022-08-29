import argparse
from utils import *
from dataset import *
from distiller_zoo import *
import models
# import sss_optimizer.modified_optim as optim_sss
import torch.optim as optim
from tqdm import tqdm
from models.utils import *
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.nn as nn
from time import time
import wandb

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--random_seed_t', default=1, type=int)
    parser.add_argument('--random_seed_s', default=2, type=int)
    parser.add_argument('--data_dir', default=r'data', type=str)
    parser.add_argument('--data', default='CIFAR100', type=str)

    # scheduling
    parser.add_argument('--epoch', default=240, type=int)
    parser.add_argument('--schedule', default=[150, 180, 210], type=int, nargs='+')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--lambda1', default=0.005, type=float)
    parser.add_argument('--lambda2', default=0.01, type=float)

    # distillation
    parser.add_argument('--distill', default='att', type=str)
    parser.add_argument('--alpha', default=0.9, type=float, help='weight for KD (Hinton)')
    parser.add_argument('--beta', default=0, type=float, help='weight for other KD')
    parser.add_argument('--gamma', default=0.1, type=float, help='weight CE')
    parser.add_argument('--temperature', default=4, type=float)
    parser.add_argument('--feat_dim', default=128, type=int)

    # model
    parser.add_argument('--model_t', default='resnet110', type=str)
    parser.add_argument('--model_s', default='resnet110', type=str)
    parser.add_argument('--use_depth', default='True', type=str2bool)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--remain_block_num', default=0, type=int)

    parser.add_argument('--is_wandb', default='True', type=str2bool)
    parser.add_argument('--wandb_project', default='resnet50_s_8blocks', type=str)

    args = parser.parse_args()
    np.random.seed(args.random_seed_s)
    torch.manual_seed(args.random_seed_s)
    torch.cuda.manual_seed(args.random_seed_s)

    # path_log = os.path.join(DATASET_PATH, 'public/mingi-ji/logs_mask')
    path_log = 'logs'
    args_path = '%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s' % (args.model_s, args.lambda1, args.lambda2, args.optimizer, args.use_depth, args.data, args.distill, args.model_t, args.gamma, args.alpha, args.beta)
    path_log = os.path.join(path_log, args_path)

    # path_load = os.path.join(DATASET_PATH, 'public/mingi-ji/trained')
    path_load = 'trained'
    args_load = 'teacher_%s_%s/%d.pth' % (args.model_t, args.data, args.random_seed_t)
    path_load = os.path.join(path_load, args_load)

    # save student
    path_save = 'trained_student'
    args_save = args.wandb_project
    path_save = os.path.join(path_save, args_save)

    if args.is_wandb:
        wandb.init(project=args.wandb_project, config=args, name=args_path, entity="jiwooshin")
        
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    path_log = path_log + '/%s.txt' % (args.random_seed_s)

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    path_save = path_save + '/%s.pth' % (args.random_seed_s)

    logger = create_logging(path_log)
    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # args.data_dir = os.path.join(DATASET_PATH, 'public/mingi-ji', args.data_dir)
    train_loader, test_loader, args.num_classes, args.num_sample_class = create_loader(args.batch_size, args.data_dir,
                                                                                       args.data)

    kwargs_t = {}
    if args.data == 'IMAGENET':
        kwargs_t['pretrained'] = True

    model_t = models.__dict__[args.model_t](num_classes=args.num_classes, **kwargs_t)

    if args.data != 'IMAGENET':
        model_t.load_state_dict(torch.load(path_load))

    # studnet model
    model_s = models.__dict__[args.model_s](num_classes=args.num_classes)

    # model_s = nn.DataParallel(model_s)
    
    device = torch.device('cuda')

    # check flops and params
    if args.data.lower() == 'cifar100':
        img_size = 32
    elif args.data == 'imagenet':
        img_size = 224
    # teacher_flops, teacher_params = check_params(model_t, img_size)
    # student_flops, student_params = check_params(model_s, img_size)
    # logger.info(f'Student flops: {student_flops}, Student params: {student_params}, Teacher flops: {teacher_flops}, Teacher params: {teacher_params}')
    

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    # module_list -> (model_s, (regress_s), model_t)
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    # trainable_list -> (model_s, (regress_s))
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = DistillKL(args.temperature)

    if args.distill == 'att':
        criterion_kd = Attention()
    elif args.distill == 'fit':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[-3].shape, feat_t[-3].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    else:
        criterion_kd = DistillKL(args.temperature)

    criterion = nn.ModuleList([])
    criterion.append(criterion_ce)
    criterion.append(criterion_kl)
    criterion.append(criterion_kd)

    name_list = []
    for name, _ in trainable_list.named_parameters():
        name_list.append(name)
    
    if args.optimizer == 'sss':
        optimizer = optim_sss.SGD_APG(trainable_list.parameters(), name_list, lambda_ = args.lambda_, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(trainable_list.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 

    module_list.append(model_t)
    module_list.cuda()
    criterion.cuda()
    cudnn.benchmark = True

    print('Training starts !!!')

    best_accuracy = 0

    if args.is_wandb:
        wandb.watch(model_s, criterion_kl, log='all', log_freq=100)


    for epoch in tqdm(range(1, args.epoch + 1)):
        s = time()
        adjust_learning_rate(optimizer, epoch, args)

        if args.use_depth:
            # resnet cifar
            depth_weight_list = [1.5, 1, 0.5]
            # resnet v2
            # depth_weight_list = [1.3, 1.1, 0.9, 0.7]
        else:
            depth_weight_list = []

        train_loss, train_acc1, train_acc5 = train_kl(module_list, optimizer, criterion, train_loader, device, args, epoch, depth_weight_list)
        test_acc1, test_acc5, avg_batch_infer_time = test(model_s, test_loader, device)

        close_gate = []
        gate_sum = 0
        for name, param in trainable_list.named_parameters():
            if 'gate' in name:
                gate_param = gate_function(param)
                gate_sum += gate_param

                if epoch % 20 == 0:
                    if args.is_wandb:
                        wandb.log({name: gate_param}, step=epoch)
                        
        if args.is_wandb:
            wandb.log({'gate_sum': gate_sum}, step=epoch)
        
        if test_acc1 > best_accuracy:
            best_accuracy = test_acc1

        logger.info(
            'Epoch: {0:2d} |Train Loss: {1:2.4f} |Train Top1: {2:.4f} |Train Top5: {3:.4f} |Test Top1: {4:.4f} |Test Top5: {5:.4f}|Best Accuracy: {6:.4f} |Time: {7:>4.0f}|Average Batch Inference Time: {8:.10f}'.format(
                epoch, train_loss, train_acc1, train_acc5, test_acc1, test_acc5, best_accuracy, time() - s, avg_batch_infer_time))
        if args.is_wandb:
            wandb.log({'epoch': epoch, 'loss': train_loss, 'train_acc': train_acc1, 'test_acc': test_acc1}, step=epoch)
    
    torch.save(model_s.state_dict(), path_save)

if __name__ == '__main__':
     main()