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
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument("--eval", action='store_true',
                        help="flag for doing evaluation")
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

## test on validation dataset and prepare for visualization
def eval_valDataset(args, model, loader,save_dir):
    mean_correct = []
    results = []
    print(len(loader))
    for j, data in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):  
        # target is the gt, points is the partial
        label, pos_observed, target = data
        pos_observed = torch.Tensor(pos_observed)
        # print(points.shape)
        target = torch.Tensor(target)
        # print("target")
        # print(target.shape)
        pos_observed, target = pos_observed.cuda(), target.cuda()
        # print("input")
        # print(points.shape)
        pred = model(pos_observed)

        # pos_observed is the first partial point cloud in this batch
        pos_observed = pos_observed.cpu().detach().numpy().reshape(-1, args.num_pts_observed, 3)[0]
        pred = pred.cpu().detach().numpy()[0]
        np.save(os.path.join(save_dir, 'pos_observed_' + str(label) + str(j)), pos_observed)
        np.save(os.path.join(save_dir, 'pred_' + str(label) + str(j)), pred)
        # if j == 10:
        #     break
        #  use get_loss function here
        # criterion(pred.view(args.bsize, -1, 3), points.view(-1, args.num_pts, 3)).mean()
        # loss = criterion(pred, points.view(-1, args.num_pts, 3)).mean()
        # print("pred")
        # print(pred.shape)
        # print(target.shape)
        # results.append(criterion(pred, target.view(-1, args.num_pts, 3)))

    # results = torch.cat(results, dim=0).mean().item()
    # logger.add_scalar('test_chamfer_dist', results, epoch)
    # print('Epoch: {:03d}, Test Chamfer: {:.4f}'.format(epoch, results))

    # results = -results



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

    # SERVER TRAINING: num_workers = 8

    TEST_DATASET = Completion3DDataset(root=DATA_PATH, class_choice=None, split='val')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=1)
    print("data len")
    print(len(testDataLoader))
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
    # model = torch.nn.DataParallel(model) # ???
    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth') 
    checkpoint = torch.load('./log/classification/2020-12-06_03-14/checkpoints/best_model.pth')
    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    # if not os.path.isfile(checkpoint):
    #     raise ValueError('{} does not exist. Please provide a valid path for pretrained model!'.format(checkpoint))
    # model.load_state_dict(torch.load(checkpoint))
    log_string('Use pretrain model')

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

    '''TESTING'''
    logger.info('Start testing...')
    save_dir = './vis_res'
    with torch.no_grad():
        eval_valDataset(args,model.eval(), testDataLoader, save_dir)

    logger.info('End of testing...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
