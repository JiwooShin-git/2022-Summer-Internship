import argparse
from utils import *
from dataset import *
import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchvision

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--random_seed', default=10, type=int)
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

    # model
    parser.add_argument('--model', default='resnet44', type=str)

    args = parser.parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # path_log = os.path.join(DATASET_PATH, 'public/mingi-ji/logs_mask')
    path_log = 'logs'
    args_path = f'teacher_{args.model}_{args.data}'
    path_log = os.path.join(path_log, args_path)
    
    # path_save = os.path.join(DATASET_PATH, 'public/mingi-ji/trained')
    path_save = 'trained'
    args_save = f'teacher_{args.model}_{args.data}'
    path_save = os.path.join(path_save, args_save)

    if not os.path.exists(path_log):
        os.makedirs(path_log)
    path_log = path_log + '/%s.txt' % (args.random_seed)

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    path_save = path_save + '/%s.pth' % (args.random_seed)

    logger = create_logging(path_log)
    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # args.data_dir = os.path.join(DATASET_PATH, 'public/mingi-ji', args.data_dir)
    train_loader, test_loader, args.num_classes, args.num_sample_class = create_loader(args.batch_size, args.data_dir, args.data)    
    model = models.__dict__[args.model](num_classes=args.num_classes)

    # check flops and params
    if args.data.lower() == 'cifar100':
        img_size = 32
    elif args.data == 'imagenet':
        img_size = 224
    flops, params = check_params(model, img_size)
    logger.info(f'flops: {flops}, params: {params}')
    
     
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.cuda()
    cudnn.benckmark = True
    device = torch.device('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, 0.0)

    best_acc = 0
    for epoch in tqdm(range(1, args.epoch + 1)):
        # scheduler.step()
        adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc1, train_acc5 = train(model, optimizer, criterion, train_loader, device)
        test_acc1, test_acc5, avg_batch_infer_time = test(model, test_loader, device)

        if test_acc1 > best_acc:
            best_acc = test_acc1

        logger.info('Epoch: {0:>2d} |Train Loss: {1:>2.4f} |Train Top1: {2:.4f} |Train Top5: {3:.4f} |Test Top1: {4:.4f} |Test Top5: {5:.4f}|Best Accuracy: {6:.4f} |Average Batch Inference Time: {7:.10f}'
                    .format(epoch, train_loss, train_acc1, train_acc5, test_acc1, test_acc5, best_acc, avg_batch_infer_time))

    torch.save(model.state_dict(), path_save)

if __name__ == '__main__':
    main()