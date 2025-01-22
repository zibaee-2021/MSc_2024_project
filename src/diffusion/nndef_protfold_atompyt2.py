#!~/miniconda3/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from math import floor, log, log2


from einops import rearrange

SIGDATA = 16
VARDATA = 256


# Symmetric ALiBi relative positional bias
class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> () h () ()')
        self.register_buffer('slopes', slopes, persistent=False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** floor(log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, residx):
        relative_position = residx.unsqueeze(0) - residx.unsqueeze(1)
        bias = torch.abs(relative_position).unsqueeze(0).unsqueeze(0).expand(1, self.heads, -1, -1)
        #  bias = torch.abs(relative_position).clip(max=40).unsqueeze(0).unsqueeze(0).expand(1, self.heads, -1, -1)
        return bias * -self.slopes


# Implementation for tied multi-head attention with ALiBi relative pos encoding
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None):
        super().__init__()
        if k_dim is None:
            k_dim = d_model
        if v_dim is None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scaling = self.d_k ** -0.25

        self.to_query = nn.Linear(d_model, d_model, bias=False)
        self.to_key = nn.Linear(k_dim, d_model, bias=False)
        self.to_value = nn.Linear(v_dim, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, posbias=None, return_att=False):
        B, L = query.shape[:2]

        q = self.to_query(query).view(B, L, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # (B, h, l, k)
        k = self.to_key(key).view(B, L, self.heads, self.d_k).permute(0, 2, 3, 1).contiguous()  # (B, h, k, l)
        v = self.to_value(value).view(B, L, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # (B, h, l, k)

        # Scale both Q & K to help avoid fp16 overflows
        q = q * self.scaling
        k = k * self.scaling
        attention = torch.einsum('bhik,bhkj->bhij', q, k)
        if posbias is not None:
            attention = attention + posbias
        attention = F.softmax(attention, dim=-1)  # (B, h, L, L)
        #
        out = torch.matmul(attention, v)  # (B, h, L, d_k)
        # print(out)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return self.to_out(out)
    

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)
    

class SeqEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, p_drop=0.1):
        super().__init__()

        # Multi-head attention
        self.attn = MultiheadAttention(d_model, heads)

        self.gate = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate.weight, 0.)
        nn.init.constant_(self.gate.bias, 1.)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 8, bias=False),
            SwiGLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model * 4, d_model, bias=False)
        )

        nn.init.zeros_(self.ff[3].weight)

        # Normalization and dropout modules
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, bias):
        # Input shape for multi-head attention: (BATCH, NRES, EMB)
        x2 = x
        x = self.norm1(x)
        x = torch.sigmoid(self.gate(x)) * self.attn(x, x, x, bias) # Tied attention over L (requires 4D input)
        x = x2 + self.dropout(x)
        
        # feed-forward
        x2 = x
        x = self.norm2(x)
        x = self.ff(x)
        return x2 + x


class AtomEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, p_drop=0.1):
        super().__init__()

        # Multi-head attention projections
        self.to_query = nn.Linear(d_model, d_model, bias=False)
        self.to_key = nn.Linear(d_model, d_model, bias=False)
        self.to_value = nn.Linear(d_model, d_model, bias=False)

        self.d_k = d_model // heads
        # Scale both Q & K to help float16 training
        self.scaling = self.d_k ** -0.25

        self.gate = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate.weight, 0.)
        nn.init.constant_(self.gate.bias, 1.)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            SwiGLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model*2, d_model)
        )

        nn.init.zeros_(self.ff[3].weight)
        nn.init.zeros_(self.ff[3].bias)

        # Normalization and dropout modules
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        # Input shape for multi-head attention: (BATCH, NRES, EMB)
        x2 = x
        x = self.norm1(x)
        q = self.to_query(x) * self.scaling
        k = self.to_key(x) * self.scaling
        v = self.to_value(x)
        attn = F.scaled_dot_product_attention(q, k, v, scale=1)
        x = torch.sigmoid(self.gate(x)) * attn

        x = x2 + self.dropout(x)
        
        # feed-forward
        x2 = x
        x = self.norm2(x)
        x = self.ff(x)
        return x2 + x
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-log(10000.0) / self.d_model))
        self.register_buffer('div_term', div_term, persistent=False)

    def forward(self, positions):
        positions = positions.unsqueeze(1).float()
        
        pe = torch.empty(positions.size(0), self.d_model, device=positions.device)
        pe[:, 0::2] = torch.sin(positions * self.div_term)
        pe[:, 1::2] = torch.cos(positions * self.div_term)
        
        return pe


class FourierEncodingLayer(nn.Module):
    def __init__(self, out_dim, in_dim=8):
        super().__init__()

        self.in_dim = in_dim
        self.out_emb = nn.Linear(in_dim*2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        device, dtype, orig_x = x.device, x.dtype, x
        scales = 2 ** torch.arange(self.in_dim, device=device, dtype=dtype)
        x = x / scales
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = self.norm(self.out_emb(x))
        return x
    

# DiffusionNet Module
class DiffusionNet(nn.Module):
    def __init__(self, seqwidth=1024, atomwidth=256, seqheads=16, atomheads=8, seqdepth=8, atomdepth=10, cycles=3):
        # increase atomwidth because protein has more atom types (38) than RNA..?
        super().__init__()

        self.cycles = cycles

        self.alibi = AlibiPositionalBias(seqheads)

        self.norm1 = nn.LayerNorm(seqwidth)
        
        layers = []
        for _ in range(seqdepth):
            layer = SeqEncoderLayer(seqwidth, seqheads, p_drop=0.1)
            layers.append(layer)
        self.seqencoder = nn.ModuleList(layers)

        self.to_coords = nn.Linear(seqwidth, 3, bias=False)
        self.to_confs = nn.Linear(seqwidth, 1, bias=False)

        self.to_atom = nn.Linear(seqwidth, atomwidth, bias=False)
        self.norm2 = nn.LayerNorm(atomwidth)
        self.norm3 = nn.LayerNorm(atomwidth)
        
        # self.nt_embed = nn.Embedding(5, atomwidth)  # 5 is number of nucleotides + 1 (presumably for unknown?)
        NUM_OF_AMINO_ACIDS = 20
        self.aa_embed = nn.Embedding(NUM_OF_AMINO_ACIDS, atomwidth)  # CHANGE TO 20

        # self.atom_embed = nn.Embedding(28, atomwidth)  # 28 is number of nucleotide atoms.
        NUM_OF_AA_ATOMS = 38
        self.atom_embed = nn.Embedding(NUM_OF_AA_ATOMS, atomwidth)  # 38 is number of nucleotide atoms.

        # self.ntidx_embed = PositionalEncoding(atomwidth)
        self.aaidx_embed = PositionalEncoding(atomwidth)

        self.coord_embed = nn.Linear(3, atomwidth, bias=False)

        self.nlev_embed = FourierEncodingLayer(atomwidth)
        
        layers = []
        for _ in range(atomdepth):
            layer = AtomEncoderLayer(atomwidth, atomheads, p_drop=0.0)
            layers.append(layer)
        self.atomencoder = nn.Sequential(*layers)

        self.out_denoise_vecs = nn.Sequential(
            nn.LayerNorm(atomwidth),
            nn.Linear(atomwidth, 3, bias=False)
        )

    #       network(inputs, aacodes, atomcodes, aaindices, noised_coords, noise_levels) # from line 371 in main()
    # def forward(self, x, ntcodes, atcodes, ntindices, noised_coords_in, nlev_in):
    def forward(self, x, aacodes, atcodes, aaindices, noised_coords_in, nlev_in):

        B, L = x.shape[0:2]

        posbias = self.alibi(torch.arange(L, device=x.device))

        x = self.norm1(x)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(self.cycles):
                for m in self.seqencoder:
                    x = m(x, posbias)

        pred_coords = self.to_coords(x)
        pred_confs = self.to_confs(x).squeeze(-1)
        
        x = self.norm2(self.to_atom(x))

        # Expand sequence embedding to atoms
        # atom_x = x[:,ntindices]
        atom_x = x[:, aaindices]

        coordscale = (nlev_in.view(-1, 1, 1).pow(2) + VARDATA).sqrt()
        
        atom_x = (atom_x
                  + self.aa_embed(aacodes)[aaindices].unsqueeze(0)
                  + self.atom_embed(atcodes).unsqueeze(0)
                  + self.aaidx_embed(aaindices).unsqueeze(0)
                  + self.nlev_embed(nlev_in).unsqueeze(1)
                  + self.coord_embed(noised_coords_in / coordscale))

        atom_x = self.norm3(atom_x)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            atom_x = checkpoint_sequential(self.atomencoder, 3, atom_x)

        t_h = nlev_in.view(-1, 1, 1)
        pred_denoised = (self.out_denoise_vecs(atom_x) * SIGDATA * t_h / (VARDATA + t_h ** 2).sqrt()
                         + noised_coords_in * VARDATA / (VARDATA + t_h ** 2))

        return pred_denoised, pred_coords, pred_confs
