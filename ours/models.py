import torch
import numpy as np
import sys
import torch.nn.functional as F
from model_utils import *
from pointnet_utlis import *

class get_model(torch.nn.Module):
    """
    Point clouds completion model.

    Arguments:
        radius: float, radius for generating sub point clouds
        bottleneck: int, the size of bottleneck
        num_vote_train: int, the number of votes during training
        num_contrib_vote_train: int, the maximum number of selected votes during training
        num_vote_test: int, the number of votes during test
    """

    def __init__(
		self,radius,bottleneck,
		num_pts,num_pts_observed,
        num_vote_train, num_contrib_vote_train,
        num_vote_test):
        super(get_model,self).__init__()
        self.num_vote_train = num_vote_train
        self.num_contrib_vote_train = num_contrib_vote_train
        self.num_vote_test = num_vote_test
        self.bottleneck = bottleneck

        self.encoder = Encoder(radius, bottleneck, self.num_vote_train, self.num_vote_test)
        self.latent_module = LatentModule(is_vote = True)
        self.decoder = FoldingBasedDecoder(bottleneck)

    def forward(self, x=None, pos=None):
        print("x in get_model forward")
        print(x.shape)
        # extract feature for each sub point clouds
        mean, std = self.encoder(x, pos)

        # select contribution features
        if self.training: # Boolean inherited from nn.Module representing whether this module is in training or evaluation mode
            contrib_mean, contrib_std = \
                self.feature_selection(mean, std, self.num_vote_train, self.num_contrib_vote_train)
        else:
            contrib_mean, contrib_std = \
                self.feature_selection(mean, std, self.num_vote_test)

        # compute optimal latent feature
        optimal_z = self.latent_module(contrib_mean, contrib_std)

        self.contrib_mean, self.contrib_std = contrib_mean, contrib_std
        self.optimal_z = optimal_z

        # generate prediction
        pred = self.decoder(optimal_z)
        return pred

    def feature_selection(
            self,
            mean,
            std,
            num_vote,
            num_contrib_vote=None,
        ):
        """
        During training, random feature selection is adapted. Only a portion of
        features generated from local point clouds will be considered for optimal
        latent feature computation. During test, all generated features will be
        contributed to calculate the optimal latent feature.
        Arguments:
            mean: [-1, bottleneck], computed mean for each sub point cloud
            std: [-1, bottleneck], computed std for each sub point cloud
            num_vote: int, the total number of extracted features from encoder.
            num_contrib_vote: int, maximum number of candidate features comtributing
                                    to optimal latent features during training
        Returns:
            new_mean: [bsize, num_contrib_vote, f], selected contribution means
            new_std: [bsize, num_contrib_vote, f], selected contribution std
        """
        mean = mean.view(-1, num_vote, mean.size(1))
        std = std.view(-1, num_vote, std.size(1))

        # feature random selection
        if self.training:
            num = np.random.choice(np.arange(1, num_contrib_vote+1), 1, False)
            idx = np.random.choice(mean.size(1), num, False)
        else:
            idx = np.arange(num_vote)
        new_mean = mean[:, idx, :]
        new_std = std[:, idx, :]

        # build a mapping
        # source_idx = torch.arange(mean.size(0)*mean.size(1))
        # target_idx = torch.arange(new_mean.size(0)*new_mean.size(1))
        # source_idx = source_idx.view(-1, num_vote)[:, idx].view(-1)
        # mapping = dict(zip(source_idx.numpy(), target_idx.numpy()))

        return new_mean, new_std

class Encoder(torch.nn.Module):
    def __init__(self, radius, bottleneck, ncentroid_train, ncentroid_test):
        super(Encoder, self).__init__()
        self.npoint = None
        if self.training:
            self.npoint = ncentroid_train
        else:
            self.npoint = ncentroid_test

        self.nsample = 128
        self.in_channel = 3
        # 3 -> 64
        # self.sa_3_64 = PointNetSetAbstraction(self.npoint, radius, 
        #                                         self.nsample, self.in_channel,
        #                                         mlp = [64], group_all = False)
        # 64+3 -> 64 -> 128 -> 512

        # 3 -> 64 -> 128 -> 512
        self.sa_module = PointNetSetAbstraction(self.npoint, radius, 
                                                self.nsample, self.in_channel,
                                                mlp = [64, 128, 512], group_all = False)
        self.mlp = customized_mlp([512+3, 512, bottleneck*2], last=True, leaky=True)
    
    def forward(self, x, points):
        # points could be None
        B = x.shape[0]
        # new_points: [B, 512, npoint]
        # new_xyz: [B, 3, npoint],coordinate of the centroids

        print("x in encoder forward")
        print(x.shape)

        new_xyz, new_points = self.sa_module(x, points)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        # new_points: [B, npoint, 512]
        # new_xyz: [B, npoint, 3]

        # after cat: [B, npoint, 512+3]
        print(new_points.shape) # [2, 64, 512]
        print(new_xyz.shape) # [2, 64, 3]
        tmp = torch.cat([new_points,new_xyz], dim=-1)
        print("tmp.shape")
        print(tmp.shape) # [2, 64, 515]

        x = self.mlp(tmp.view(-1,515))
        mean, logvar = torch.chunk(x, 2, dim=-1)
        logvar = torch.clamp(logvar, min=-100, max=100)
        # std must be positive
        std = torch.exp(0.5*logvar)
        return mean, std

class LatentModule(torch.nn.Module):
    def __init__(self, is_vote):
        super(LatentModule, self).__init__()
        self.is_vote = is_vote

    def forward(self, mean, std):
        """
        mean: [bsize, n, bottleneck]
        """
        # guassian model to get optimal
        if self.is_vote:
            denorm = torch.sum(1/std, dim=1)
            nume = torch.sum(mean/std, dim=1)
            optimal_z = nume / denorm       # [bsize, bottleneck]
        # max pooling
        else:
            optimal_z = mean.max(dim=1)[0]
            # optimal_z = mean.mean(dim=1)
        return optimal_z

class FoldingBasedDecoder(torch.nn.Module):
    def __init__(self, bottleneck):
        """
        Same decoder structure as proposed in the FoldingNet
        """
        super(FoldingBasedDecoder, self).__init__()
        self.fold1 = FoldingNetDecFold1(bottleneck)
        self.fold2 = FoldingNetDecFold2(bottleneck)

    def forward(self, x):  # input x = batch, 512
        batch_size = x.size(0)
        x = torch.unsqueeze(x, 1)  # x = batch,1,512
        x = x.repeat(1, 45 ** 2, 1)  # x = batch,45^2,512
        code = x
        code = x.transpose(2, 1)  # x = batch,512,45^2

        meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
        grid = GridSamplingLayer(batch_size, meshgrid)  # grid = batch,45^2,2
        grid = torch.from_numpy(grid)

        if x.is_cuda:
            grid = grid.cuda()

        x = torch.cat((x, grid), 2)  # x = batch,45^2,514
        x = x.transpose(2, 1)  # x = batch,514,45^2

        x = self.fold1(x)  # x = batch,3,45^2
        x = torch.cat((code, x), 1)  # x = batch,515,45^2
        x = self.fold2(x)  # x = batch,3,45^2

        return x.transpose(2, 1)


class FoldingNetDecFold1(torch.nn.Module):
    def __init__(self, bottleneck):
        super(FoldingNetDecFold1, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck+2, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 3, 1)
        self.relu = torch.nn.ReLU()
        # self.bn1 = torch.nn.BatchNorm1d(512)
        # self.bn2 = torch.nn.BatchNorm1d(512)

    def forward(self, x):  # input x = batch,514,45^2
        x = self.relu(self.conv1(x))  # x = batch,512,45^2
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        # x = self.relu(self.bn1(self.conv1(x)))  # x = batch,512,45^2
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.conv3(x)
        return x


class FoldingNetDecFold2(torch.nn.Module):
    def __init__(self, bottleneck):
        super(FoldingNetDecFold2, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck+3, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 3, 1)
        self.relu = torch.nn.ReLU()
        # self.bn1 = torch.nn.BatchNorm1d(512)
        # self.bn2 = torch.nn.BatchNorm1d(512)

    def forward(self, x):  # input x = batch,515,45^2
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        # x = self.relu(self.bn1(self.conv1(x)))  # x = batch,512,45^2
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.conv3(x)
        return x

def GridSamplingLayer(batch_size, meshgrid):
    """
    output Grid points as a NxD matrix
    params = {
    'batch_size': 8
    'meshgrid': [[-0.3,0.3,45],[-0.3,0.3,45]]
    }
    """
    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    grid = np.zeros((np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32)  # MxD
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
    return g
    
class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    
    def forward(self, x, y):
        """
        Compute chamfer distance for x and y. Note there are multiple version of chamfer
        distance. The implemented chamfer distance is defined in:
            https://arxiv.org/pdf/1612.00603.pdf.
        It finds the nearest neighbor in the other set and computes their squared
        distances which are summed over both target and ground-truth sets.
        Arguments:
            x: [bsize, m, 3]
            y: [bsize, n, 3]
        Returns:
            dis: [bsize]
        """
        # x: [bsize,1, m, 3]
        x = x.unsqueeze(1)
        # y: [bsize, n,1, 3]
        y = y.unsqueeze(2)
        # x,y: [bsize,n, m, 3]
        diff = (x - y).norm(dim=-1)
        # diff = (x - y).pow(2).sum(dim=-1)
        dis1 = diff.min(dim=1)[0].mean(dim=1)
        dis2 = diff.min(dim=2)[0].mean(dim=1)
        dis = dis1 + dis2
        return dis