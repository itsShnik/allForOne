# --------------------------------------------------------
# All For One: Multi-modal Multi-task Learning
# Written by Nikhil Shah https://github.com/itsShnik
# --------------------------------------------------------

from openvqa.models.allForOne.adapter import Adapter
from openvqa.models.allForOne.attentionTeams import AttentionTeams
import torch.nn as nn
import torch.nn.functional as F
import torch

# ----------------------------
# ---- All for one model -----
# ----------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.adapter = Adapter(__C)
        self.backbone = AttentionTeams(__C)

        # Classification layers
        self.proj = nn.Linear(__C.HIDDEN_SIZE, answer_size)


    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)

        img_feat = self.adapter(frcn_feat, grid_feat, bbox_feat)

        # sum the img_feats
        img_feat = img_feat.sum(1).unsqueeze(1)

        # Backbone Framework
        lang_feat = self.backbone(
            lang_feat,
            img_feat
        )

        # Classification layers
        proj_feat = self.proj(lang_feat)

        return proj_feat

