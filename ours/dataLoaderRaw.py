#!/usr/bin/env python
# coding: utf-8

# In[1]:


# *_*coding:utf-8 *_*
import os
import json
import h5py
import warnings
import numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


# In[48]:


import torch
class EPN3D__class(Dataset):
    
    def __init__(self, root, class_choice=None, split='train'):
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.categories = self.catfile
        self.cat = {}
        self.allFilenames = []
        self.path = []
        self.data = []
        
        # open file and load the categories
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
#         print(self.cat)
        
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
#             print(filenamesTotal[0])
#             print(len(filenamesTotal))
            return filenamesTotal
    
        def getData(filenamesT, splitInGet):
            gtList = []
            partialList = []
            gtCatList = []
            for name in filenamesT:
                cat = name.split("/")[0]
                
                fpos = None
                pos = None
                # if the input split is 'train', we only use the ground truth for training
                if splitInGet == 'train':
                    fpos = h5py.File(os.path.join(self.path[0], name), 'r')
                    pos = torch.tensor(fpos['data'], dtype=torch.float32)
                    gtList.append(pos)
                    gtCatList.append(cat)
                
                fy = None
                y = None

                if splitInGet == 'val':
                    fpos = h5py.File(os.path.join(self.path[0], name), 'r')
                    pos = torch.tensor(fpos['data'], dtype=torch.float32)
                    fy = h5py.File(os.path.join(self.path[1], name), 'r')
                    y = torch.tensor(fy['data'], dtype=torch.float32)
                    partialList.append(pos)
                    gtList.append(y)
                    gtCatList.append(cat)
            
            if splitInGet == 'train':
                print(len(gtList))
                print(len(gtCatList))
                return [gtList, gtCatList]
            if splitInGet == 'val':
                dataList = [partialList, gtList, gtCatList]
                return dataList             
                               
        
        # get the file
        def process_names(split_in_loop):
            data_list = []
            categories_ids = self.cat.values()
#             print(categories_ids)

            # get the train data
            if split_in_loop == 'train':
                self.path.append(os.path.join(self.root, 'train', 'gt'))
#                 self.allFilenames.append(getFileNames(path[0]))
                self.allFilenames = getFileNames(self.path[0])
#             print(len(self.allFilenames))

            # get the val data gt
            if split_in_loop == 'val':
                # get the val data partial
                self.path.append(os.path.join(self.root, 'val', 'partial'))
#                 self.allFilenames.append(getFileNames(path[0]))
                self.allFilenames = getFileNames(self.path[0])
                # for val need to do twice in the following part
                self.path.append(os.path.join(self.root, 'val', 'gt'))
#                 self.allFilenames.append(getFileNames(self.path[1]))
            
            return getData(self.allFilenames, split_in_loop)
            
        if split == "train":
            print("here")
            self.data = process_names("train")
#             print("self.data len: ", len(data))
        elif split == "val":
            self.data = process_names("val")

        

    def __getitem__(self, index):
        print("self.data len: ", len(self.data))
        return self.data
        
    def __len__(self):
#         print()
        return len(self.data[-1])


# In[ ]:





# In[49]:


res = EPN3D__class("C:/Users/Minzhe/Desktop/EECS545/FinalProject/shapenet")


# In[50]:


print(len(res))


# In[57]:


trainP = res[0]
print(trainP[0][3].shape)
print(type(trainP))

