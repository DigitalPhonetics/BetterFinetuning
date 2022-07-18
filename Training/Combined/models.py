"""
Adapt from:
https://github.com/facebookresearch/barlowtwins/blob/main/main.py
"""
import torch
import torch.nn as nn
from transformers import HubertModel
from transformers import Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

import Training.config as config


def off_diagonal(x):
    """
    For the purpose of calculation:
    return flattened view of the off-diagonal elements of a square matrix
    """
    n, m = x.shape
    # need to ensure it is matrix
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, output_size, lambd, batch_size, device):
        super().__init__()
        self.output_size = output_size
        self.lambd = lambd
        self.batch_size = batch_size
        self.device = device

        # linear layer as projector
        # self.linear_layer = nn.Sequential(nn.Linear(1024, 64))
        self.dropout = nn.Sequential(nn.Dropout(0.5))
        if config.backbone == "wav2vec2":
            self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.backbone.fc = nn.Identity()
        elif config.backbone == "hubert":
            self.backbone = HubertModel.from_pretrained("facebook/hubert-base")
            self.backbone.fc = nn.Identity()
        else:
            import sys
            print("Invalid backbone selected!")
            sys.exit()
        # We will try to use projector in the original paper
        # 3-layers projector
        proj_layers = []
        for layer in range(3):
            if layer == 0:  # first layer
                proj_layers.append(nn.Linear(1024, self.output_size, bias=False))
            else:
                proj_layers.append(
                    nn.Linear(self.output_size, self.output_size, bias=False)
                )
            if layer < 2:  # if not the last layer
                proj_layers.append(nn.BatchNorm1d(self.output_size))
                proj_layers.append(nn.ReLU(inplace=True))
        self.projector = nn.Sequential(*proj_layers)
        self.bn = nn.BatchNorm1d(self.output_size, affine=False)

    def forward(self, input_1, input_2):
        # compute masked indices
        batch_size, raw_sequence_length = input_1.shape
        sequence_length = self.backbone._get_feat_extract_output_lengths(
            raw_sequence_length
        )
        mask_time_indices = _compute_mask_indices(
            (batch_size, sequence_length), mask_prob=0.2, mask_length=2
        )
        mask_time_indices = torch.from_numpy(mask_time_indices).to(self.device)

        # compute masked indices
        n = input_1.shape[0]
        # print("n: \n", n) # 32
        output_1 = self.backbone(
            input_1, mask_time_indices=mask_time_indices
        ).extract_features  # [32, 2, 512]
        output_1 = output_1.reshape(n, -1)  # [32, 1024]
        # TODO: try droupout
        output_1 = self.dropout(output_1)
        # print("output_1: \n", output_1.shape) # 32

        # TODO: (batch)normalization version of representation
        # output_1 = self.linear_layer(output_1)  # [32, 64]
        output_1 = self.projector(output_1)

        output_2 = self.backbone(
            input_2, mask_time_indices=mask_time_indices
        ).extract_features
        # TODO: remove reshape perphas
        output_2 = output_2.reshape(n, -1)
        # output_2 = self.linear_layer(output_2)
        output_2 = self.projector(output_2)
        # TODO: try droupout
        output_2 = self.dropout(output_2)

        return output_1, output_2

    def loss(self, output_1, output_2):
        # empirical cross-correlation matrix
        c = self.bn(output_1).T @ self.bn(output_2)  # [32, 64]

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)  # 32 is batch size
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss_val = on_diag + self.lambd * off_diag
        return loss_val


class BarlowTwins_Contrastive(nn.Module):
    def __init__(
            self, output_size, lambd, triplet_margin, barlowtwins_lambd, batch_size, device
    ):
        super().__init__()
        self.output_size = output_size
        self.lambd = lambd
        self.barlowtwins_lambd = barlowtwins_lambd
        self.batch_size = batch_size
        self.device = device
        self.cosine_similarity = nn.CosineSimilarity()
        self.triplet_margin = triplet_margin

        # linear layer as projector
        # self.linear_layer = nn.Sequential(nn.Linear(1024, 64))
        self.dropout = nn.Sequential(nn.Dropout(0.5))
        if config.backbone == "wav2vec2":
            self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        elif config.backbone == "hubert":
            self.backbone = HubertModel.from_pretrained("facebook/hubert-base")
        else:
            import sys
            print("Invalid backbone selected!")
            sys.exit()
        # self.backbone.fc = nn.Identity()
        # 3-layers projector
        proj_layers = []
        for layer in range(3):
            if layer == 0:  # first layer
                proj_layers.append(nn.Linear(1024, self.output_size, bias=False))
            else:
                proj_layers.append(
                    nn.Linear(self.output_size, self.output_size, bias=False)
                )
            if layer < 2:  # if not the last layer
                proj_layers.append(nn.BatchNorm1d(self.output_size))
                proj_layers.append(nn.ReLU(inplace=True))
        self.projector = nn.Sequential(*proj_layers)
        self.bn = nn.BatchNorm1d(self.output_size, affine=False)

    def forward(self, anchor, positive, negative):
        # compute masked indices
        n = anchor.shape[0]
        batch_size, raw_sequence_length = anchor.shape
        sequence_length = self.backbone._get_feat_extract_output_lengths(
            raw_sequence_length
        )
        mask_time_indices = _compute_mask_indices(
            (batch_size, sequence_length), mask_prob=0.2, mask_length=2
        )
        mask_time_indices = torch.from_numpy(mask_time_indices).to(self.device)

        anchor_out = self.backbone(
            anchor, mask_time_indices=mask_time_indices
        ).extract_features
        anchor_out = self.dropout(anchor_out)
        anchor_out = anchor_out.reshape(n, -1)
        anchor_out = self.projector(anchor_out)

        positive_out = self.backbone(
            positive, mask_time_indices=mask_time_indices
        ).extract_features
        positive_out = self.dropout(positive_out)
        positive_out = positive_out.reshape(n, -1)
        positive_out = self.projector(positive_out)

        negative_out = self.backbone(
            negative, mask_time_indices=mask_time_indices
        ).extract_features
        negative_out = self.dropout(negative_out)
        negative_out = negative_out.reshape(n, -1)
        negative_out = self.projector(negative_out)

        return anchor_out, positive_out, negative_out

    def barlowtwins_loss(self, anchor_out, positive_out):
        # empirical cross-correlation matrix
        c = self.bn(anchor_out).T @ self.bn(positive_out)  # [32, 64]

        # sum the cross-correlation matrix between all gpus
        # TODO: use argueparser for batch size 32
        c.div_(self.batch_size)  # 32 is batch size
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss_val = on_diag + self.barlowtwins_lambd * off_diag
        return loss_val

    def triplet_loss(self, anchor_out, positive_out, negative_out, reduction="mean"):
        positive_distance = 1 - self.cosine_similarity(anchor_out, positive_out)

        negative_distance = 1 - self.cosine_similarity(anchor_out, negative_out)

        losses = torch.max(
            positive_distance - negative_distance + self.triplet_margin,
            torch.full_like(positive_distance, 0),
        )
        if reduction == "mean":
            return torch.mean(losses)
        else:
            return torch.sum(losses)

    def combine_loss(self, barlowtwins_loss, triplet_loss):
        return barlowtwins_loss * self.lambd + triplet_loss
