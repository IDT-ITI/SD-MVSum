# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def pairwise_cosine_similarity(A, B):
    A_norm = F.normalize(A, p=2, dim=1)  # (N, D)
    B_norm = F.normalize(B, p=2, dim=1)  # (M, D)
    return torch.mm(A_norm, B_norm.T)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_pos=512):
        """
        Class implementing sinusoidal absolute positional encoding for sequence data.
        :param int dim: Dimensionality of the embeddings (should be even).
        :param int max_pos: Maximum sequence length to support. Defaults to 512.
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
        Generates positional encoding for a sequence of given length.
        :param int n: The number of positions (i.e., sequence length) to generate encodings for.
        :param torch.device device: The device to move the positional encoding tensor to. Defaults to CUDA.
        :return torch.Tensor: A tensor of shape [n, dim] containing the positional encodings.
        """
        enc = self.pe[:n]
        return enc.to(device)


class CrossAttention(nn.Module):
    def __init__(self, input_size=512, text_size=512, output_size=512, heads=8, pos_enc=True, similarity_weight=False):
        """
        Class implementing multi-head (language-guided) cross-attention between video and text features
        :param int input_size: Dimensionality of the input video features.
        :param int text_size: Dimensionality of the input text features.
        :param int output_size: Dimensionality of the hidden/output space for each head. Defaults to 512.
        :param int heads: Number of attention heads. Defaults to 8.
        :param bool pos_enc: Whether to apply sinusoidal positional encoding. Defaults to True.
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
        Compute multi-head cross-attention between video and text inputs.

        :param torch.Tensor video_features: Input video features with shape [N, input_size], where N is the number of frames.
        :param torch.Tensor text_features: Text feature tensor with shape [M, text_size], where M is the number of sentences.
        :return torch.Tensor: Output video features with shape [N, input_size] after attention.
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
    model = CrossAttention().cuda()
    video_features = torch.randn(500, 512).cuda()  # [num_frames, hidden_size]
    text_features = torch.randn(7, 512).cuda()  # [num_sentences, hidden_size]
    output, weights = model(video_features, text_features)
    print(f"Output shape: {output.shape}")
    """