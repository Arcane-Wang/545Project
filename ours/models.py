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

        ratio_train: float, sampling ratio in further points sampling (FPS) during training
        ratio_test: float, sampling ratio in FPS during test

        num_vote_train: int, the number of votes during training
        num_contrib_vote_train: int, the maximum number of selected votes during training
        num_vote_test: int, the number of votes during test

    """

    def __init__(
		self,radius,bottleneck,
		num_pts,num_pts_observed, m
        num_vote_train,num_contrib_vote_train,num_vote_test, 
        ratio_train,ratio_test):
      super(Module,self).__init__()
      self.num_vote_train = num_vote_train
      self.num_contrib_vote_train = num_contrib_vote_train
      self.num_vote_test = num_vote_test
      self.task = task
      self.bottleneck = bottleneck

      # selected ncentroid number of centroids
      ncentroid_train = int(ratio_train * num_pts)
	  ncentroid_test = int(ratio_test * num_pts_observed)

	  self.encoder = Encoder(radius, bottleneck, ncentroid_train, ncentroid_test)
	  self.latent_module = LatentModule()
	  self.decoder = FoldingBasedDecoder(bottleneck)

    def forward(self, x=None, pos=None, batch=None, category=None):
        batch = batch - batch.min()

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
        self.sa_module = SAModule(radius, ncentroid_train, ncentroid_test,
                                  mlp([64+3, 64, 128, 512], leaky=True))
        self.mlp = mlp([512+3, 512, bottleneck*2], last=True, leaky=True)

    def forward(self, x, pos, batch):
        x, new_pos, new_batch, x_idx, y_idx = self.sa_module(x, pos, batch)
        x = self.mlp(torch.cat([x, new_pos], dim=-1))
        mean, logvar = torch.chunk(x, 2, dim=-1)
        # in case gradient explodes
        # logvar = torch.clamp(logvar, min=-100, max=100)
        std = torch.exp(0.5*logvar)
        # self.new_pos = new_pos
        return mean, std, x_idx, y_idx

class SAModule(torch.nn.Module):
    def __init__(self, r, ncentroid_train, ncentroid_test, nn):
        """ 
        Set abstraction module, which is proposed by Pointnet++.
        r: ball query radius
        ncentroid_train: number of sampling points in further points sampling (FPS) during training.
        ncentroid_test: number of sampling points in FPS during test.
        nn: mlp
        """
        super(SAModule, self).__init__()
        self.r = r
        self.ncentroid_train = ncentroid_train
        self.ncentroid_test = ncentroid_test
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        if self.training:
            ncentroid = self.ncentroid_train
        else:
            ncentroid = self.ncentroid_test
        idx = fps(pos, batch, ncentroid=ncentroid)
        # ball query searches neighbors
        y_idx, x_idx = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=128)

        edge_index = torch.stack([x_idx, y_idx], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch, x_idx, y_idx



