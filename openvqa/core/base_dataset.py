# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import numpy as np
import glob, json, torch, random
import torch.utils.data as Data
import torch.nn as nn
from torchvision import transforms
from openvqa.utils.feat_filter import feat_filter

class BaseDataSet(Data.Dataset):
    def __init__(self):
        self.token_to_ix = None
        self.pretrained_emb = None
        self.ans_to_ix = None
        self.ix_to_ans = None

        self.data_size = None
        self.token_size = None
        self.ans_size = None

        self.normalize = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                std=[0.229, 0.224, 0.225])

        self.transform = transforms.compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            self.normalize,
            ])


    def load_ques_ans(self, idx):
        raise NotImplementedError()


    def load_img_feats(self, idx, iid):
        raise NotImplementedError()


    def __getitem__(self, idx):

        ques_ix_iter, ans_iter, iid = self.load_ques_ans(idx)

        img_iter, frcn_feat_iter, grid_feat_iter, bbox_feat_iter = self.load_img_feats(idx, iid)

        # transform the image first
        # already converts the image to tensor, so no need to do so later on
        img_iter = self.transform(img_iter)

        return \
            img_iter,\
            torch.from_numpy(frcn_feat_iter),\
            torch.from_numpy(grid_feat_iter),\
            torch.from_numpy(bbox_feat_iter),\
            torch.from_numpy(ques_ix_iter),\
            torch.from_numpy(ans_iter)


    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)


class BaseAdapter(nn.Module):
    def __init__(self, __C):
        super(BaseAdapter, self).__init__()
        self.__C = __C
        if self.__C.DATASET in ['vqa']:
            self.vqa_init(__C)

        elif self.__C.DATASET in ['gqa']:
            self.gqa_init(__C)

        elif self.__C.DATASET in ['clevr']:
            self.clevr_init(__C)

        else:
            exit(-1)

        # eval('self.' + __C.DATASET + '_init()')

    def vqa_init(self, __C):
        raise NotImplementedError()

    def gqa_init(self, __C):
        raise NotImplementedError()

    def clevr_init(self, __C):
        raise NotImplementedError()

    def forward(self, frcn_feat, grid_feat, bbox_feat):
        feat_dict = feat_filter(self.__C.DATASET, frcn_feat, grid_feat, bbox_feat)

        if self.__C.DATASET in ['vqa']:
            return self.vqa_forward(feat_dict)

        elif self.__C.DATASET in ['gqa']:
            return self.gqa_forward(feat_dict)

        elif self.__C.DATASET in ['clevr']:
            return self.clevr_forward(feat_dict)

        else:
            exit(-1)

    def vqa_forward(self, feat_dict):
        raise NotImplementedError()

    def gqa_forward(self, feat_dict):
        raise NotImplementedError()

    def clevr_forward(self, feat_dict):
        raise NotImplementedError()

