import itertools
import os
import time
import random
import argparse
import datetime
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import util.subset_sampler as subsetsampler
from config import get_config
from models import build_model
from util.logger import create_logger
from util.utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper
from util.learning_scheme import build_scheduler
import torch.optim as optim
import torchvision.transforms as transforms
import util.process_datas as pd
from torch.utils.data import DataLoader
from loss.asy_loss import *
from loss.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss, corrective_triplet_loss
from evaluate import calc_map
from models.awnet import AWNet
from loss.loss_new import NCEandMAE,NCEandRCE
from tqdm import tqdm
import math as m
import torch.nn.functional as F
from save_mat import Save_mat

import  os
os.environ['CUDA_LAUNCH_BLOCKING']  =  "1"

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None




def parse_option():
    parser = argparse.ArgumentParser('Deep Global Semantic Structure-preserving Hashing', add_help=False)
    parser.add_argument('--cfg', type=str, default='./configs/swin_config.yaml', metavar="FILE",
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default='./dataset/UCMD/', help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    #./pretrained/vit_small_patch16_224.pth
    parser.add_argument('--pretrained', default='./pretrained/vit_small_patch16_224.pth',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')

    parser.add_argument('--resume',help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, default=2, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='./result/', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--triplet_strategy', default='batch_corrective', type=str,
                        help='triplet_strategy')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default= 50)
    # parser.add_argument('--n_class', type=int, default=19)
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def encoding_onehot(target, nclasses=30):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

import numpy as np

def sample_weight(output, target):
    output_sort = output.argsort(descending=True)
    output_sort_list = output_sort.cpu().numpy().tolist()
    target_list = target.tolist()
    weight = []
    # print(len(output_sort_list), len(output_sort_list[0]))
    # print(output_sort_list)
    # print(target_list)
    for ele in output_sort_list:
        index = output_sort_list.index(ele)
        if output[index][ele[0]] == output[index][target_list[index]]:
            pn = output[index][ele[1]]
        else:
            pn = output[index][ele[0]]
        delta = output[index][target_list[index]] - pn
        weight.append(np.float64(delta))

    return weight





def _dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dset_database = pd.ProcessingUCMD_21(
        'dataset/UCMD', 'database_index_img.txt', 'database_index_label.txt', transformations)
    dset_test = pd.ProcessingUCMD_21(
        'dataset/UCMD', 'test_index_img.txt', 'test_index_label.txt', transformations)
    num_database, num_test = len(dset_database), len(dset_test)

    def load_label(filename, DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        fp = open(path, 'r')
        labels = [x.strip() for x in fp]
        fp.close()
        return torch.LongTensor(list(map(int, labels)))

    testlabels = load_label('test_index_label.txt', 'dataset/UCMD')
    databaselabels = load_label('database_index_label.txt', 'dataset/UCMD')

    testlabels = encoding_onehot(testlabels,nclasses=config.MODEL.NUM_CLASSES)
    databaselabels = encoding_onehot(databaselabels,nclasses=config.MODEL.NUM_CLASSES)

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)
    return nums, dsets, labels

def main(config):
    # TODO data_loader_database; tensorboard
    '''
     dataset preprocessing
     '''
    nums, dsets, labels = _dataset()
    # n_class = nums
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    args, config = parse_option()

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    model.train()
    start = time.time()
    logger.info(str(model))


    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        select_index = list(np.random.permutation(range(num_database)))[0: num_database]
        _sampler = subsetsampler.SubsetSampler(select_index)
        trainloader = DataLoader(dset_database, batch_size=32,
                                 sampler=_sampler,
                                 shuffle=False,
                                 num_workers=0)
        '''
                       ...
                        '''

        for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
            optimizer.zero_grad()  # 将梯度归0
            batch_size_ = train_label.size(0)
            u_ind = np.linspace(iteration * 32, np.min((num_database, (iteration + 1) * 32)) - 1,
                                batch_size_, dtype=int)

            train_input = Variable(train_input.cuda())
            train_label = train_label.cuda()
            train_label = train_label.squeeze(1)
        '''
                        training procedure finishes, evaluation
                        '''

        for k in range(config.MODEL.hash_length):
            sel_ind = np.setdiff1d([ii for ii in range(config.MODEL.hash_length)], k)
            V_ = V[:, sel_ind]
            Uk = U[:, k]
            U_ = U[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))
        loss_ = calc_loss(output, V, U,select_index,config.MODEL.beta_param, train_label,config.MODEL.NUM_CLASSES)

        logger.info('[epoch: %3d][Train Loss: %.4f]', epoch, loss_)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))

        model.eval()
        testloader = DataLoader(dset_test, batch_size=1,
                                shuffle=False,
                                num_workers=0)

        '''
                       ...
                        '''

        if mAP > Best_mAP:

            Best_mAP = mAP
            l = qB.shape[1]
            # print(test_labels.shape, zong_labels.shape)
            Save_mat(epoch=epoch + 1, output_dim=l, datasets=config.DATA.DATASET,
                     query_labels=test_labels.numpy(),
                     retrieval_labels=database_labels.numpy(),
                     query_img=qB,
                     retrieval_img=rB, save_dir='.',
                     mode_name="DGSSH",
                     mAP=Best_mAP)
        # logger.info(
            # f"{config.MODEL.info} epoch:{epoch + 1} bit:{config.MODEL.hash_length} dataset:{config.DATA.DATASET} MAP:{mAP} Best MAP: {Best_mAP}")
        logger.info('[Evaluation: mAP: %.4f]', Best_mAP)
if __name__ == '__main__':


    main()



