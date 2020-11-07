import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def index_points(points, idx):
    """

    Input:
        (xyz is [B,N,3])
        points: input points data, [B, N, C]

        (fps_idx is [B, npoint], every batch has npoint number of centroids)
        idx: sample index data, [B, S]
        
    Return:
        new_points:, indexed points data, [B, S, C]
        (i.e. [B,npoint,3])
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # for every batch, we will generate npoint numbers of centroids
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # there are B*N numbers of points in total
    # record the distance from points to centorids set
    # will be updated every time a new centorid is added to the centorids set
    distance = torch.ones(B, N).to(device) * 1e10
    # generate an initial cenrtoid for every batch -> shape is (B,)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # 0,1,2 ... B
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
 
    # npoint loops, find npoint centorids for every batch
    for i in range(npoint):
        centroids[:, i] = farthest
        # the coordinate of the newest centroid
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # dist to the newest centroid in its batch
        # (x-xc1)^2 + (y-yc1)^2 + (z-zc1)^2
        # dim=-1, every point have one corresponding dist value
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # update the distance value, 
        # distance = min(old_distance, distance to the newest centroid)
        mask = dist < distance
        distance[mask] = dist[mask]
        # farthest's shape : (B,) 
        # every loop will find the ith centroid for every batch
        farthest = torch.max(distance, -1)[1]
    return centroids

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: number of centroids for every batch
        radius: local region radius
        nsample: max sample number in local region
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3] ?????
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # fps_idx shape: [B, npoint]
    fps_idx = farthest_point_sample(xyz, npoint) 
    torch.cuda.empty_cache()

    # new_xyz shape: [B, npoint, 3]
    new_xyz = index_points(xyz, fps_idx) 
    torch.cuda.empty_cache()

    # idx shape: [B, npoint, nsample]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()

    # grouped_xyz shape: [B, npoint, nsample, 3]
    grouped_xyz = index_points(xyz, idx) # 
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

	return TODO

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

