import torch
import numpy as np
import sys
import torch.nn.functional as F
from .model_utils import *

class Model(torch.nn.Module):
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
      super(Module,self).__init__()
      self.num_vote_train = num_vote_train
      self.num_contrib_vote_train = num_contrib_vote_train
      self.num_vote_test = num_vote_test
      self.bottleneck = bottleneck

	  self.encoder = Encoder(radius, bottleneck, self.num_vote_train, self.num_vote_test)
	  self.latent_module = LatentModule(is_vote = True)
	  self.decoder = FoldingBasedDecoder(bottleneck)

    def forward(self, x=None, pos=None):
        #x = self.transformer(pos)
        # extract feature for each sub point clouds
        mean, std, x_idx, y_idx = self.encoder(x, pos, batch)

        # select contribution features
        if self.training:
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
        self.sa_3_64 = PointNetSetAbstraction(self.npoint, radius, 
                                                self.nsample, self.in_channel,
                                                mlp = [64], group_all = False)
        # 64+3 -> 64 -> 128 -> 512
        self.sa_module = PointNetSetAbstraction(self.npoint, radius, 
                                                self.nsample, self.in_channel,
                                                mlp = [64, 128, 512], group_all = False)
        self.mlp = customized_mlp([512+3, 512, bottleneck*2], last=True, leaky=True)
    
    def forward(self, x, points):
        # points could be None
        # new_points: [B, 512, npoint]
        # new_xyz: [B, 3, npoint],coordinate of the centroids
        new_xyz_1, new_points_1 = self.sa_3_64(x, points)
        new_xyz, new_points = self.sa_module(new_xyz_1, new_points_1)

        # add controid coordinate after xyz
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        # new_points: [B, npoint, 512]
        # new_xyz: [B, npoint, 3]
        # after cat: [B, npoint, 512+3]
        x = self.mlp(torch.cat([new_points,new_xyz], dim=-1))
        mean, logvar = torch.chunk(x, 2, dim=-1)
        std = torch.exp(0.5*logvar)
        return mean, std

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

