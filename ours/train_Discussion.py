
"""
Author: Benny
Date: Nov 2019
"""
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import importlib
import shutil
from models import *
from completion3D_dataset import Completion3DDataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():  # Set up parameters to input
    '''PARAMETERS'''
    # parser = argparse.ArgumentParser('PointNet')
    # parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    # parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    # parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    # parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    
    # parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    # parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    # parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    # return parser.parse_args()

    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    
    parser.add_argument("--model_name", default='model',
                        help="model name")
    parser.add_argument("--dataset", type=str, choices=['shapenet', 'modelnet', 'completion3D', 'scanobjectnn'],
                        help="shapenet or modelnet or completion3D")
    parser.add_argument("--categories", default='Chair',
                        help="point clouds categories, string or [string]. For ShapeNet: Airplane, Bag, \
                        Cap, Car, Chair, Earphone, Guitar, Knife, Lamp, Laptop, Motorbike, Mug, Pistol, \
                        Rocket, Skateboard, Table; For Completion3D: plane;cabinet;car;chair;lamp;couch;table;watercraft")
    parser.add_argument("--task", type=str, choices=['completion', 'classification', 'segmentation'],
                        help=' '.join([
                            'completion: point clouds completion',
                            'classification: shape classification on ModelNet40',
                            'segmentation: part segmentation on ShapeNet'
                        ]))
    parser.add_argument("--num_pts", type=int,
                        help="the number of input points")
    parser.add_argument("--num_pts_observed", type=int,
                        help="the number of points in observed point clouds")  
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="batch size")
    parser.add_argument("--step_size", type=int, default=200,
                        help="step size to reduce lr")
    parser.add_argument("--max_epoch", type=int, default=500,
                        help="max epoch to train")
    parser.add_argument("--bsize", type=int, default=32,
                        help="batch size")
    parser.add_argument("--radius", type=float,
                        help="radius for generating sub point clouds")
    parser.add_argument("--bottleneck", type=int,
                        help="the size of bottleneck")
    parser.add_argument("--num_vote_train", type=int, default=64,
                        help="the number of votes (sub point clouds) during training")
    parser.add_argument("--num_contrib_vote_train", type=int, default=10,
                        help="the maximum number of contribution votes during training")
    parser.add_argument("--num_vote_test", type=int,
                        help="the number of votes (sub point clouds) during test")
    parser.add_argument("--is_vote", action='store_true',
                        help="flag for computing latent feature by voting, otherwise max pooling")
    parser.add_argument('--model', default='models', help='model name [default: pointnet_cls]')
    return parser.parse_args()

# ## test dataset
# def evaluation():

#     # sampling in the latent space to generate diverse prediction
#     latent = model.module.optimal_z[0, :].view(1, -1)
#     idx = np.random.choice(args.num_vote_test, 1, False)
#     random_latent = model.module.contrib_mean[0, idx, :].view(1, -1)
#     random_latent = (random_latent + latent) / 2
#     pred_diverse = model.module.generate_pc_from_latent(random_latent)

#     ## save as the format that visualizer needs (numpy file n*3)
#     #
#     #
#     #

## validation dataset
def test_one_epoch(model, loader,criterion, epoch):
    mean_correct = []
    results = []
    # for j, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
    for j, data in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):  
        # target is the gt, points is the partial
        _, points, target = data
        points = torch.Tensor(points)
        # print(points.shape)
        target = torch.Tensor(target)
        # print("target")
        # print(target.shape)
        points, target = points.cuda(), target.cuda()
        # print("input")
        # print(points.shape)
        pred = model(points)

        #  use get_loss function here
        # criterion(pred.view(args.bsize, -1, 3), points.view(-1, args.num_pts, 3)).mean()
        # loss = criterion(pred, points.view(-1, args.num_pts, 3)).mean()
        # print("pred")
        # print(pred.shape)
        # print(target.shape)
        results.append(criterion(pred, target.view(-1, args.num_pts, 3)))

    results = torch.cat(results, dim=0).mean().item()
    # logger.add_scalar('test_chamfer_dist', results, epoch)
    print('Epoch: {:03d}, Test Chamfer: {:.4f}'.format(epoch, results))
    results = -results

    return results


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    assert args.dataset in ['completion3D']
    assert args.task in ['completion']
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = './data_root/shapenet'

    TRAIN_DATASET = Completion3DDataset(root=DATA_PATH, class_choice=None, split='train')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.bsize, shuffle=True, num_workers=1)
    # SERVER TRAINING: num_workers = 8

    TEST_DATASET = Completion3DDataset(root=DATA_PATH, class_choice=None, split='val')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.bsize, shuffle=True, num_workers=1)

    '''MODEL LOADING'''
    # MODEL = importlib.import_module(args.model)
    shutil.copy('./%s.py' % args.model, str(experiment_dir))  
    shutil.copy('./model_utils.py', str(experiment_dir))

    model = get_model(
        radius=args.radius,
        bottleneck=args.bottleneck,
        num_pts=args.num_pts,
        num_pts_observed=args.num_pts_observed,
        num_vote_train=args.num_vote_train,
        num_contrib_vote_train=args.num_contrib_vote_train,
        num_vote_test=args.num_vote_test,
    ) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    criterion = get_loss().to(device)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')  
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.2)

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_acc = -100
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.max_epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.max_epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = torch.Tensor(points)
            # print(points.shape)

            points = points.cuda()
            optimizer.zero_grad()

            model = model.train()
            pred = model(points)
            # print("pred shape train")
            # print(pred.shape)
            # compare the prediction and points(gt)
            # mean() : get the mean of [bsize]
            loss = criterion(pred, points.view(-1, args.num_pts, 3)).mean()
            # whether to use mean().item() ???
            log_string('Train Current Accuracy: %f' % loss)

            loss.backward()
            optimizer.step()
            global_step += 1

        ################# call test_one_epoch to get the acc for val dataset ###########
        with torch.no_grad():
            acc = test_one_epoch(model.eval(), testDataLoader, criterion, epoch)

            log_string('Test Accuracy: %f'% (acc))
            log_string('Best Accuracy: %f'% (best_acc))

            if acc > best_acc:
                # save model
                #### set the path to save model there ###
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model.pth'))
                best_acc = acc

            # if (instance_acc >= best_instance_acc):
            #     logger.info('Save model...')
            #     savepath = str(checkpoints_dir) + '/best_model.pth'
            #     log_string('Saving at %s'% savepath)
            #     state = {
            #         'epoch': best_epoch,
            #         'instance_acc': instance_acc,
            #         'class_acc': class_acc,
            #         'model_state_dict': classifier.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }
            #     torch.save(state, savepath)

            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
