# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.layers.attention import CrossAttention


class SD_MVSum(nn.Module):
    def __init__(self, input_size=512, text_size=512, output_size=512, pos_enc=True,
                 heads=8, visual_weights=False, transcript_weights=False):
        """
        SD_MVSum: A sequence-to-frame importance scoring model using multi-head
        cross-attention between video frames, transcripts, and text features.
        Args:
            input_size (int): Feature dimensionality for video frames or transcript embeddings.
            text_size (int): Feature dimensionality for text features.
            output_size (int): Hidden dimensionality for cross-attention outputs.
            pos_enc (bool): Whether to apply sinusoidal positional encoding. Defaults to True.
            heads (int): Number of attention heads in each CrossAttention module. Defaults to 8.
            visual_weights (bool): Whether to apply similarity weighting in the visual cross-attention. Defaults to False.
            transcript_weights (bool): Whether to apply similarity weighting in the transcript cross-attention. Defaults to False.
        """
        super(SD_MVSum, self).__init__()


        self.cross_attention_visual = CrossAttention(input_size=input_size, text_size=text_size, output_size=output_size,
                                        pos_enc=pos_enc, heads=heads, similarity_weight=visual_weights)
        self.cross_attention_transcript = CrossAttention(input_size=input_size, text_size=text_size, output_size=output_size,
                                        pos_enc=pos_enc, heads=heads, similarity_weight=transcript_weights)
        self.linear_layer_dim_reduction = nn.Linear(in_features=2*input_size, out_features=input_size)
        self.linear_layer = nn.Linear(in_features=input_size, out_features=1)
        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.sigmoid = nn.Sigmoid()

        # =========== Transformer ===========
        transformer_kwargs = {
            'batch_first': False,
        }
        self.transformer = nn.Transformer(input_size, **transformer_kwargs)


    def forward(self, frame_features, text_features, transcripts):
        """
        Produce frame importance scores using the SD_MVSum model.
          Args:
            frame_features (torch.Tensor): Tensor of shape [N, input_size] representing video frame embeddings, where N is the number of frames.
            text_features (torch.Tensor): Tensor of shape [M, text_size] representing text embeddings, where M is the number of sentences.
            transcripts (torch.Tensor): Tensor of shape [T, input_size] representing transcript embeddings, where T is the number of transcript segments.
          Returns:
             torch.Tensor: Tensor of shape [1, N] containing frame importance scores in [0, 1].
        """
        # ========= Cross-Attention =============

        cross_visual = self.cross_attention_visual(frame_features, text_features)
        cross_text = self.cross_attention_transcript(transcripts, text_features)
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
    """
    Uncomment for a quick proof of concept
    model = SD_VMSum().cuda()
    frame_features = torch.randn(500, 512).cuda()       # [num_frames, hidden_size]
    text_features = torch.randn(7, 512).cuda()          # [num_sentences, hidden_size]
    transcript_features = torch.randn(20, 512).cuda()   # [num_transcripts, hidden_size]
    output = model(frame_features, text_features, transcript_features)
    print(f"Output shape: {output.shape}")  # [1, 500]
    
    """
