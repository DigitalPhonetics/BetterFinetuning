import torch
from torch import nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    _compute_mask_indices,
)

from ContrastiveLearning.TripletLoss import TripletLoss


class TripletLossNet(nn.Module):
    def __init__(self, output_size, margin, device):
        super().__init__()

        self.device = device
        self.output_size = output_size
        # output_sizes = [self.output_size*3, self.output_size*2, self.output_size]
        # alternative, we can use 3-layer projector (better result?)
        # 3-layers projector
        proj_layers = []
        for layer in range(3):
            if layer == 0:  # first layer
                proj_layers.append(nn.Linear(512, self.output_size, bias=False))
            else:
                proj_layers.append(
                    nn.Linear(self.output_size, self.output_size, bias=False)
                )
            if layer < 2:  # if not the last layer
                proj_layers.append(nn.BatchNorm1d(self.output_size))
                proj_layers.append(nn.ReLU(inplace=True))
        self.projector = nn.Sequential(*proj_layers)
        self.bn = nn.BatchNorm1d(self.output_size, affine=False)

        self.dropout = nn.Sequential(nn.Dropout(0.5))
        self.rnn = nn.RNN(
            input_size=512, hidden_size=256, num_layers=1, batch_first=True
        )
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.loss_layer = TripletLoss(margin)
        self.linear_layer = nn.Sequential(nn.Linear(512, self.output_size))

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
            )

        return outputs

    def forward(self, anchor, positive, negative):

        # compute masked indices
        def _get_mask_time_indices(embedding):
            batch_size, raw_sequence_length = embedding.shape
            sequence_length = self.wav2vec_model._get_feat_extract_output_lengths(
                raw_sequence_length
            )
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length), mask_prob=0.2, mask_length=2
            )
            mask_time_indices = torch.from_numpy(mask_time_indices).to(self.device)
            return mask_time_indices

        # batch_size = anchor.shape[0]
        anchor_out = self.wav2vec_model(
            anchor, mask_time_indices=_get_mask_time_indices(anchor)
        ).extract_features
        anchor_out = self.dropout(anchor_out)
        anchor_out = self.merged_strategy(
            hidden_states=anchor_out
        )  # = outputs.last_hidden_state ?
        anchor_out = self.projector(anchor_out)

        positive_out = self.wav2vec_model(
            positive, mask_time_indices=_get_mask_time_indices(positive)
        ).extract_features
        positive_out = self.dropout(positive_out)
        positive_out = self.merged_strategy(hidden_states=positive_out)
        positive_out = self.projector(positive_out)

        negative_out = self.wav2vec_model(
            negative, mask_time_indices=_get_mask_time_indices(negative)
        ).extract_features
        negative_out = self.dropout(negative_out)
        negative_out = self.merged_strategy(hidden_states=negative_out)
        negative_out = self.projector(negative_out)

        return anchor_out, positive_out, negative_out

    def loss(self, anchor, positive, negative, reduction="mean"):
        loss_val = self.loss_layer(anchor, positive, negative, reduction)
        return loss_val
