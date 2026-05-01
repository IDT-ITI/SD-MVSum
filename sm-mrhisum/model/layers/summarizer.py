# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.layers.attention import CrossAttention


class SD_MVSum(nn.Module):
    def __init__(self, input_size=512, text_size=512, output_size=512, pos_enc=True,
                 heads=8, visual_weights=True, transcript_weights=True):
        """
        Class wrapping the SD-MVSum model; its key modules and parameters.
        :param int input_size: The expected input feature size for video features
        :param int text_size: The expected input feature size for text features.
        :param int output_size: The hidden feature size of the attention mechanisms.
        :param bool pos_enc: Whether to apply sinusoidal positional encoding.
        :param int heads: The number of attention heads.
        :param visual_weights (bool): Whether to apply similarity weighting in the visual cross-attention. Defaults to False.
        :param transcript_weights (bool): Whether to apply similarity weighting in the transcript cross-attention. Defaults to False.
        """
        super(SD_MVSum, self).__init__()

        self.cross_attention_visual = CrossAttention(input_size=input_size, text_size=text_size, output_size=output_size,
                                        pos_enc=pos_enc, heads=heads, similarity_weight=visual_weights)
        self.cross_attention_transcript = CrossAttention(input_size=input_size, text_size=text_size, output_size=output_size,
                                                         pos_enc=pos_enc, heads=heads, similarity_weight=transcript_weights)
        self.linear_layer_dim_reduction = nn.Linear(in_features=2 * input_size, out_features=input_size)
        self.linear_layer = nn.Linear(in_features=input_size, out_features=1)
        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.sigmoid = nn.Sigmoid()

        # =========== Transformer ===========
        transformer_kwargs = {
            'batch_first': False,
        }
        self.transformer = nn.Transformer(input_size, **transformer_kwargs)


    def forward(self, frame_features, text_features, transcript_features):
        """
        Produce frame importance scores using the SD_MVSum model.
        :param torch.Tensor frame_features: Tensor of shape [N, input_size] containing frame features, where N is the number of frames.
        :param torch.Tensor text_features: Tensor of shape [M, text_size] containing text features, where M is the number of sentences.
        :param torch.Tensor transcript_features: Tensor of shape [N, input_size] containing transcript features, where N is the number of frames.
        :return torch.Tensor: Tensor of shape [1, N] containing the frame importance scores in [0, 1].
        """
        # ========= Weighted Cross-Attention =============
        cross_visual = self.cross_attention_visual(frame_features, text_features)
        cross_text = self.cross_attention_transcript(transcript_features, text_features)
        attended_values = torch.cat((cross_visual, cross_text), 1)
        attended_values = self.linear_layer_dim_reduction(attended_values)

        y = self.drop(attended_values)
        y = self.norm_y(y)

        # ========= Transformer =============
        y = self.transformer(y, y)
        y = self.linear_layer(y)
        y = self.sigmoid(y)
        y = y.view(1, -1)

        return y


if __name__ == '__main__':
    pass
