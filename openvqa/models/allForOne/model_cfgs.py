# --------------------------------------------------------
# OpenVQA
# All for One: Multi-modal Multi-tasking
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# Modified by Nikhil Shah https://github.com/itsShnik
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.LAYER = 6
        self.NUM_TEAMS = 10
        self.IMG_LINEAR_SIZE = 512
        self.WORD_EMBED_SIZE = 512
        self.HIDDEN_SIZE = 512
        self.BBOXFEAT_EMB_SIZE = 2048
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024
        self.USE_AUX_FEAT = False
        self.USE_BBOX_FEAT = False
