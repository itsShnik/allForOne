# --------------------------------------------------------
# All for one - Multi-modal multi-task learning
# Written by Nikhil Shah https://github.com/itsShnik
# --------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# ----------------------------
# ---- Additive Attention ----
# ----------------------------

class Attention(nn.Module):
    def __init__(self, __C):
        super(Attention, self).__init__()

        # initialize a linear layer to project to a single dimension
        self.linear = nn.Linear(2*__C.HIDDEN_SIZE, 1)
        self.softmax = nn.Softmax(1)

    def forward(self, feats):
        attention_weights = self.linear(feats)
        attention_weights = torch.tanh(attention_weights)
        attention_weights = self.softmax(attention_weights)
        return attention_weights


# --------------------------
# ---- Attention teams  ----
# --------------------------

class AttentionTeams(nn.Module):
    def __init__(self, __C):
        super(AttentionTeams, self).__init__()

        # initialize num_teams number of GRU units
        self.unit_list = nn.ModuleList([nn.GRU(
                input_size=__C.WORD_EMBED_SIZE,
                hidden_size=__C.HIDDEN_SIZE,
                num_layers=1, 
                batch_first=True
            ) for _ in range(__C.NUM_TEAMS)])

        # initialize a lead unit
        self.lead_unit = nn.GRU(
                input_size=__C.WORD_EMBED_SIZE,
                hidden_size=__C.HIDDEN_SIZE,
                num_layers=1, 
                batch_first=True
            )

        # initialize a self-attention module
        self.attention = Attention(__C)

    def forward(self, lang_feat, img_feat):

        # concatenate the input along objects and words
        feat = torch.cat((img_feat,lang_feat), -2)

        # create a matrix of vectors obtained using each unit
        unit_feat_list = []

        for i in range(len(self.unit_list)):
            self.unit_list[i].flatten_parameters()
            _, temp_vec = self.unit_list[i](feat)
            temp_vec = temp_vec.transpose(0,1)
            unit_feat_list.append(temp_vec)

        # (batch_size, attention_units, feature_size)
        unit_feats = torch.cat(unit_feat_list, -2)

        # lead unit features
        self.lead_unit.flatten_parameters()
        _, lead_unit_feat = self.lead_unit(feat) 
        lead_unit_feat = lead_unit_feat.transpose(0,1)

        # calculate attention weights
        lead_unit_feat = lead_unit_feat.repeat(1, len(self.unit_list), 1)

        combined_unit_feats = torch.cat((unit_feats, lead_unit_feat), -1)

        attention_weights = self.attention(combined_unit_feats)

        # calculate the weighted sum of feature vectors
        proj_feat = attention_weights * unit_feats
        proj_feat = proj_feat.sum(-2)

        return proj_feat
