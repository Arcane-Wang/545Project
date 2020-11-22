import os
import h5py
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
import random

warnings.filterwarnings('ignore')

class Completion3DDataset(Dataset):
    
    def __init__(self, root, class_choice=None, split='train'):
        self.root = root
        #self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        #self.categories = self.catfile
        #self.cat = {}
        self.allFilenames = []
        self.path = []
        
        # define the categories
        self.cat = {
            'plane': '02691156',
            'cabinet': '02933112',
            'car': '02958343',
            'chair': '03001627',
            'lamp': '03636649',
            'couch': '04256520',
            'table': '04379243',
            'watercraft': '04530566',
        }
        self.cat = {v: k for k, v in self.cat.items()}
        if not class_choice is None:
            self.cat = {k:v for k,v in self.cat.items() if v in class_choice}
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        
        def getFileNames(path):
            filenamesTotal = []
            catList = []
            with os.scandir(path) as f:
                for cat in f:
                    catList.append(cat.name)

            for cat in catList:
                filenames = []
                with os.scandir(os.path.join(path, cat)) as f1:
                    filenames = [cat+'/'+file.name for file in f1]
                filenamesTotal = filenamesTotal + filenames
            return filenamesTotal
    
        def getData(filenamesT, splitInGet):
            self.cache = []
            for name in filenamesT:
                cat = name.split("/")[0]
                
                fpos = None
                pos = None
                fy = None
                y = None
                
                # if the input split is 'train', we only use the ground truth for training
                if splitInGet == 'train':
                    # complete point cloud
                    fpos = h5py.File(os.path.join(self.path[0], name), 'r')
                    pos = torch.tensor(fpos['data'], dtype=torch.float32)
                    self.cache.append((pos, self.classes[cat]))
                
                elif splitInGet == 'val':
                    # partial point cloud
                    fpos = h5py.File(os.path.join(self.path[0], name), 'r')
                    pos = torch.tensor(fpos['data'], dtype=torch.float32)
                    # complete point cloud
                    fy = h5py.File(os.path.join(self.path[1], name), 'r')
                    y = torch.tensor(fy['data'], dtype=torch.float32)
                    self.cache.append((self.classes[cat], pos, y))    

                elif splitInGet == 'test':
                    # partial point cloud
                    fpos = h5py.File(os.path.join(self.path[0], name), 'r')
                    pos = torch.tensor(fpos['data'], dtype=torch.float32)
                    self.cache.append(pos)          
                    
        
        # get the file
        def process_names(split_in_loop):
            # get the train data
            if split_in_loop == 'train':
                self.path.append(os.path.join(self.root, 'train', 'gt'))
                self.allFilenames = getFileNames(self.path[0])

            # get the val data
            elif split_in_loop == 'val':
                # get the val data partial
                self.path.append(os.path.join(self.root, 'val', 'partial'))
                self.allFilenames = getFileNames(self.path[0])
                self.path.append(os.path.join(self.root, 'val', 'gt'))
                
            # get the test data
            elif split_in_loop == 'test':
                self.path.append(os.path.join(self.root, 'test', 'partial'))
                self.allFilenames = getFileNames(self.path[0])

            getData(self.allFilenames, split_in_loop)
            
            # random cache
            if split_in_loop == 'train':
                random.shuffle(self.cache)
            
        process_names(split)

    def __getitem__(self, index):
        return self.cache[index]
        
    def __len__(self):
        return len(self.cache)