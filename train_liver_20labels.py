import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from test_util import load_test_name_list,test_all_case

from networks_v1.vnet_sdf import VNet
#from networks_v1.discriminator import FC3DDiscriminator

from dataloaders import utils
from utils1 import ramps, losses, metrics
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler,  CreateOnehotLabel
from utils1.util import compute_sdf, mix_match

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/mnt/ai2019/jing/DTC/processed2/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/DTC_with_consis_weight', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=20, help='random seed')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float,  default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
parser.add_argument('--crop_size', type=int, default=48)
parser.add_argument('--detail', type=int,  default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1,
                    help='apply NMS post-procssing?')
args = parser.parse_args()

train_data_path = args.root_path

# ??????????????????, ??????label???beta?????????????????????????????????
snapshot_path = "../model_liver/" + args.exp + \
    "_{}labels_beta_{}".format(
        args.labelnum, args.beta)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# ???batch_size????????????????????????x????????????batch_size
batch_size = args.batch_size * len(args.gpu.split(','))
# ???????????????
max_iterations = args.max_iterations
# ???????????????(???????????????)
base_lr = args.base_lr
# ??????batch_size???, ????????????????????????????????? (?????????????????? * len(args.gpu.split(',')), ???????????????)
labeled_bs = args.labeled_bs #* len(args.gpu.split(','))

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def soft_cross_entropy(predicted, target):
    return -(target * torch.log(predicted)).sum(dim=1).mean()

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.base_lr

        # ???????????????????????????????????????????????????
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        # ???????????????ema????????????????????????
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                # 0.99 * ???????????????????????? + 0.01 * ????????????????????????
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def linear_ramp(maximum_lambda):
    '''
    ramp up from step 100 to step 200
    '''

    def func(e):
        if e < 100:
            return 0
        else:
            return min(maximum_lambda, (e - 100) * 0.01 * maximum_lambda)

    return func


def slow_linear_ramp(maximum_lambda):
    '''
    ramp up from step 100 to step 600
    '''

    def func(e):
        if e < 100:
            return 0
        else:
            return min(maximum_lambda, (e - 100) * 0.002 * maximum_lambda)

    return func


def train_main():
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    # ??????????????????
    print(snapshot_path + "/log.txt")
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # ????????????????????????????????????
    logging.info(str(args))

    def create_model(ema=False):
        model = VNet(n_channels=1, n_classes=num_classes,
                   normalization='batchnorm', has_dropout=True)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    model1 = create_model(ema=True)
    # ema model???????????????
    ema_optimizer= WeightEMA(model, model1, alpha=args.ema_decay)

    db_train = LAHeart(base_dir=train_data_path,

                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ])
) 
# ??????: batch_size x 1 x patch_size, ??????: batch_size x class x patch_size(one_hot??????)

    labelnum = args.labelnum    # default 16
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 104))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,  # ??????????????????batchsize????????????
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    model1.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()
    x_criterion = soft_cross_entropy
    Lambda_ramp = linear_ramp(1.5)
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl': # ??????
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()

        Lambda = Lambda_ramp(epoch_num)
        print(f"Lambda ramp: Lambda = {Lambda}")
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label'] # volume_batch :[4, 1, 112, 112, 80]  label_batch [4, 112, 112, 80]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            #label_batch = label_batch #.unsqueeze(1)  # [4, 2, 112, 112, 80]

            l_image = volume_batch[:labeled_bs].cpu().numpy()  # ?????????????????????????????? numpy
            l_label = label_batch[:labeled_bs].cpu().numpy()# ????????????????????????  numpy  (2, 2, 112, 112, 80)
          
            u_image = volume_batch[labeled_bs:]  # ?????????????????????????????????  tensor (2, 1, 112, 112, 80)

            X = list(zip(l_image, l_label))
            U = u_image
            # 'ww': ?????????????????????????????????????????????????????????????????????, ????????????????????????????????????????????????????????????????????????  (????????????????????????????????????)
            # 'xx': ?????????????????????????????????????????????????????????????????????????????????, ?????????????????????????????????????????????????????????????????????????????????, 
            # 'uu': ????????????????????????????????????????????????????????????????????????????????????, ????????????????????????????????????????????????????????????????????????????????????, 
            # '__': ??????????????????
            X_prime, U_prime, pseudo_label = mix_match(X, U, eval_net=model1, K=2, T=0.2, alpha=0.75, mixup_mode='__', aug_factor=1)  # ww
            
            # ??????????????????
            model.train()
            X_data = torch.from_numpy(np.array([x[0] for x in X_prime]))
            X_label = torch.from_numpy(np.array([x[1] for x in X_prime])) #[labeled_bs, 2, 112, 112, 80]
            X_label = X_label.argmax(dim=1) # [labeled_bs, 112, 112, 80] ????????????one-hot??????
            U_data = torch.from_numpy(np.array([x[0] for x in U_prime])) #[(batch_size - labeled_bs) * K, 1, 112, 112, 80]
            U_label = torch.from_numpy(np.array([x[1] for x in U_prime])) #[(batch_size - labeled_bs) * K, 2, 112, 112, 80]
            U_label = U_label.argmax(dim=1) # [(batch_size - labeled_bs) * K, 112, 112, 80] ????????????one-hot??????
            # print(U_label.shape )
            X_data = X_data.cuda()
            X_label = X_label.cuda().float()

            U_data = U_data.cuda()
            U_label = U_label.cuda().float()
            X = torch.cat((X_data, U_data), 0)
            U = torch.cat((X_label, U_label), 0)

            # 1. ???????????????????????? ???????????? Dual-task Consistency ???????????????????????????loss

            outputs_tanh, outputs = model(X)  # outputs_tanh -> (-1, 1), outputs?????????
            outputs_tanh = outputs_tanh[:, 1, ...]
            outputs_x_soft = F.softmax(outputs[:labeled_bs, ...], dim=1)[:, 1, ...]  # ??????????????? (K, 112, 112, 80)
            outputs_u_soft = F.softmax(outputs[labeled_bs:, ...], dim=1)[:, 1, ...]  # ?????????????????? (K, 112, 112, 80)
            outputs_soft = F.softmax(outputs, dim=1)  # (K, n_class, 112, 112, 80)
            outputs_soft = outputs_soft[:, 1, ...]  # (K, 112, 112, 80) ????????????1?????????
            
            # ????????????loss??????
            with torch.no_grad():
                gt_dis = compute_sdf(X_label[:].cpu().numpy(), outputs[:labeled_bs, 1, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            loss_sdf = mse_loss(outputs_tanh[:labeled_bs, ...], gt_dis)  # ????????????1?????????, ??????????????????????????????????????????????????????????????????
            loss_seg = ce_loss(outputs[:labeled_bs, 1, ...], X_label[:].float())  # ??????????????????????????????????????????????????????????????????
            loss_seg_dice = losses.dice_loss(outputs_x_soft[:labeled_bs, ...], X_label[:] )  # ?????????????????????????????????????????????????????????????????????
            #supervised_loss = 0.5 * (loss_seg_dice + loss_seg) + args.beta * loss_sdf
            supervised_loss = loss_seg + args.beta * loss_sdf
        
            # ???????????????loss??????
            dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)  # ??????????????????(??????????????????????????????), ?????????????????????????????????????????????, ??????2?????????????????????????????????????????????


            consistency_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)

            dis_to_mask_U = torch.sigmoid(-1500 * outputs_tanh[labeled_bs:, ...])

            T=0.2
            pse_dis1 = dis_to_mask ** (1 / T)
            pse_dis2 = (1 - dis_to_mask) ** (1 / T)
            pse_dis = pse_dis1 / (pse_dis1 + pse_dis2)
            unlabel_loss = torch.mean((U_label - outputs_u_soft) ** 2)
            consistency_loss = torch.mean((U - outputs_soft) ** 2) + torch.mean((dis_to_mask - outputs_soft) ** 2) + torch.mean((pse_dis - outputs_soft) ** 2)
            #consistency_loss =  torch.mean((pse_dis - outputs_soft) ** 2)+torch.mean((U - outputs_soft) ** 2)
            # ??????????????????????????????loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = loss_seg + Lambda*unlabel_loss+consistency_weight*consistency_loss
            #loss = supervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ???ema????????????????????????
            ema_optimizer.step()
            # dc = metrics.dice(torch.argmax(
            #     outputs_soft[:labeled_bs], dim=1), label_batch[:labeled_bs])

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/unlabel_loss', unlabel_loss, iter_num)
            # writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            # writer.add_scalar('loss/consistency_weight',
            # consistency_weight, iter_num)
            # writer.add_scalar('loss/consistency_loss',
            # consistency_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, unlabel_loss: %f,  loss_seg: %f' %
                (iter_num, loss.item(), unlabel_loss.item(), loss_seg.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                # volume_batch -> (b, 1, 112, 112, 80)
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # outputs_soft -> (b, 112, 112, 80)
                image = outputs_soft[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                # dis_to_mask -> (b, 112, 112, 80)
                # image = dis_to_mask[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # grid_image = make_grid(image, 5, normalize=False)
                # writer.add_image('train/Dis2Mask', grid_image, iter_num)

                # outputs_tanh -> (b, 112, 112, 80)
                # image = outputs_tanh[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # grid_image = make_grid(image, 5, normalize=False)
                # writer.add_image('train/DistMap', grid_image, iter_num)

                # label_batch -> (b, n_class, 112, 112, 80)
                image = label_batch.argmax(dim=1)[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                # gt_dis -> (b, 112, 112, 80)
                # image = gt_dis[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # grid_image = make_grid(image, 5, normalize=False)
                # writer.add_image('train/Groundtruth_DistMap', grid_image, iter_num)

            # change lr
            # ???2500????????????????????????, ??????*0.1
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            # ???1000???????????????
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
        writer.close()

def test():

    snapshot_path = '/mnt/ai2019/zxg_FZU/semi-supervised/model_liver/LA/DTC_with_consis_weight_20labels_beta_0.3/'

    num_classes = 2
    test_save_path = os.path.join(snapshot_path, "test_live/")
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    filename_list = load_test_name_list(os.path.join(args.root_path, 'test_path_list.txt'))
    net = VNet(n_channels=1, n_classes=num_classes,
               normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(
        snapshot_path, 'iter_6000.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, filename_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path,
                               metric_detail=args.detail, nms=args.nms)

    return avg_metric

if __name__ == "__main__":
    train_main()
    test()