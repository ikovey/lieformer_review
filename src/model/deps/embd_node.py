import torch

from .atomic.fn_scatter import scatter_sum
from .atomic.module_mlp import qMLP


class NeighborEmbd(torch.nn.Module):
    def __init__(self, dn, de, n_rbf):
        super().__init__()
        self.dn = dn
        self.rbf_lin = qMLP([n_rbf, dn], norm=None, activation=None)
        # self.extra_lin = qMLP([2 * dn, dn])

    def forward(self, z_embd, edge_index, rbf_basis, edge_decays):
        # suppose z: (nA, 1)
        src, dst = edge_index[0], edge_index[1]
        xjs = torch.index_select(z_embd, 0, src)  # )|E|, dn)

        # std
        rbf_w = self.rbf_lin(rbf_basis)
        e_impact = rbf_w * edge_decays
        ms = xjs * e_impact

        # Alternative radial feature construction:
        # rbf_basis = torch.concat([rbf_basis, edge_decays], dim=1)
        # rbf_w = self.rbf_lin(rbf_basis)  # (|E|, dn)
        # e_impact = rbf_w
        # ms = xjs * e_impact

        # e_impact = rbf_w * edge_decays  # (|E|, dn)
        # rbf_w = torch.nn.functional.silu(rbf_w)
        # e_impact = self.extra_lin(torch.concat([rbf_w, edge_decays.expand(-1, self.dn)], dim=-1))  # (|E|, dn)
        # ms = xjs * e_impact

        output = scatter_sum(ms, dst, dim=0, dim_size=z_embd.shape[0])  # (nA, dn)
        return output


class NodeEmbd(torch.nn.Module):
    def __init__(self, dn, de, n_rbf, max_atomic_number=100):
        super().__init__()

        self.atom_embed = torch.nn.Embedding(max_atomic_number + 1, dn)
        self.neigh_embd = NeighborEmbd(dn, de, n_rbf)
        self.atom_lin = qMLP([2 * dn, dn, dn], "ln", "silu")

    def forward(self, z, edge_index, rbf_basis, edge_decays):
        atom_embd = self.atom_embed(z)
        neigh_embd = self.neigh_embd(atom_embd, edge_index, rbf_basis, edge_decays)
        node_embd = self.atom_lin(torch.cat([atom_embd, neigh_embd], -1))
        return node_embd


class NodeHigh(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, a_lo, e_lo, r_ij, mask, neighbor_indices, neighbor_mask):
        pass
