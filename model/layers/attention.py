# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def pairwise_cosine_similarity(A, B):
    """
    Compute pairwise cosine similarity between two sets of vectors.
    Args:
        A (torch.Tensor): Tensor of shape (N, D), where N is the number of vectors and D is the embedding dimension.
        B (torch.Tensor): Tensor of shape (M, D), where M is the number of vectors and D is the embedding dimension.
    Returns:
        torch.Tensor: Cosine similarity matrix of shape (N, M), where each entry (i, j) is the cosine similarity between A[i] and B[j].
    """
    A_norm = F.normalize(A, p=2, dim=1)  # (N, D)
    B_norm = F.normalize(B, p=2, dim=1)  # (M, D)
    return torch.mm(A_norm, B_norm.T)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_pos=512):
        """
         Class implementing sinusoidal absolute positional encoding for sequence data.
         :param int dim: Dimensionality of the embeddings (should be even).
         :param int max_pos: Maximum sequence length to support. Defaults to 512.
         Notes:
             These positional encodings can be added to input embeddings before an
             attention mechanism. By doing so, they provide the model with positional
             information that affects the computation of similarity weights (e.g.,
             dot products or cosine similarity) between tokens.
         """
        super().__init__()
        pos = torch.arange(max_pos)
        freq = torch.arange(dim // 2) / dim
        freq = (freq * torch.tensor(10000).log()).exp()
        x = rearrange(pos, 'L -> L 1') / freq
        x = rearrange(x, 'L d -> L d 1')
        pe = torch.cat((x.sin(), x.cos()), dim=-1)
        self.pe = rearrange(pe, 'L d sc -> L (d sc)')

    def forward(self, n, *, device=torch.device('cuda')):
        """
            Generate positional encodings for a sequence of given length.
            :param int n: Sequence length (number of positions).
            :param torch.device device: Device to move the encoding tensor to. Defaults to CUDA.
            :return torch.Tensor: Positional encoding tensor of shape [n, dim].
         """
        enc = self.pe[:n]
        return enc.to(device)


class CrossAttention(nn.Module):
    def __init__(self, input_size=512, text_size=512, output_size=512, heads=8, pos_enc=True, similarity_weight=False):
        """
        Multi-head cross-attention between two modalities
        (e.g., frames ↔ text or transcripts ↔ text).
        :param int input_size: Dimensionality of the first modality (queries).
        :param int text_size: Dimensionality of the second modality (keys/values).
        :param int output_size: Dimensionality of the hidden/output space.
                                Should be divisible by the number of heads. Defaults to 512.
        :param int heads: Number of attention heads. Defaults to 8.
        :param bool pos_enc: Whether to apply sinusoidal positional encoding to the output.
                             Defaults to True.
        :param bool similarity_weight: If True, reweights attention scores using a pairwise
                                       cosine similarity matrix between the two modalities.
                                       Defaults to False.
        """

        super(CrossAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.heads = heads
        self.pos_enc = pos_enc
        self.similarity_weight = similarity_weight

        self.Wk, self.Wq, self.Wv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(self.heads):
            self.Wk.append(nn.Linear(in_features=text_size, out_features=output_size // heads, bias=False))
            self.Wq.append(nn.Linear(in_features=input_size, out_features=output_size // heads, bias=False))
            self.Wv.append(nn.Linear(in_features=text_size, out_features=output_size // heads, bias=False))
        self.out = nn.Linear(in_features=output_size, out_features=input_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.drop = nn.Dropout(p=0.5)

        if self.pos_enc:
            self.pe = PositionalEncoding(self.input_size, max_pos=4096)

    def forward(self, mode1, mode2):
        """
            Compute multi-head cross-attention between two input feature sequences.
            :param torch.Tensor mode1: Query tensor of shape [N, input_size],
                e.g., frame features or transcript features.
            :param torch.Tensor mode2: Key/Value tensor of shape [M, text_size],
                e.g., text features that guide attention.
            :return torch.Tensor: Tensor of shape [N, input_size] containing the attended representation of `mode1` conditioned on `mode2`.
        """

        outputs = []
        if mode2.ndim == 1:
            mode2 = mode2.unsqueeze(0)
        if mode1.ndim == 1:
            mode1 = mode1.unsqueeze(0)
        similarity_matrix = pairwise_cosine_similarity(mode1, mode2)

        for head in range(self.heads):

            K = self.Wk[head](mode2)
            Q = self.Wq[head](mode1)
            V = self.Wv[head](mode2)

            energies = torch.matmul(Q, K.transpose(1, 0))
            if self.similarity_weight:
                energies = torch.mul(energies, similarity_matrix)

            att_weights = self.softmax(energies)
            _att_weights = self.drop(att_weights)
            y = torch.matmul(_att_weights, V)

            # Save the current head output
            outputs.append(y)
        y = self.out(torch.cat(outputs, dim=1))
        if self.pos_enc:
            y+= self.pe(y.shape[0], device=y.device)
        return y


if __name__ == '__main__':
    pass
    """Uncomment for a quick proof of concept
    # Initialize model
    cross_attention_visual = CrossAttention().cuda()
    cross_attention_transcript = CrossAttention().cuda()
    
    # Example features
    video_features = torch.randn(500, 512).cuda()     # [num_frames, hidden_size]
    text_features = torch.randn(7, 512).cuda()        # [num_sentences, hidden_size]
    transcript_features = torch.randn(20, 512).cuda() # [num_transcripts, hidden_size]

    # Cross-attention: frames - text
    cross_visual = cross_attention_visual(video_features, text_features)
    print(f"Cross-visual output shape: {cross_visual.shape}")

    # Cross-attention: transcripts - text
    cross_text = cross_attention_transcript(transcript_features, text_features)
    print(f"Cross-text output shape: {cross_text.shape}")

    # Fuse outputs (as in the main model)
    attended_values = torch.cat((cross_visual, cross_text), dim=1)
    print(f"Fused attended shape: {attended_values.shape}")
    """