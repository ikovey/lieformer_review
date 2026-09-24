import torch

from .atomic.fn_scatter import scatter_sum
from .atomic.module_mlp import qMLP


class EdgeEmbd(torch.nn.Module):
    def __init__(self, dn, de, n_rbf):
        super().__init__()

        self.rbf_lin = qMLP([n_rbf, dn], norm=None, activation=None)
        # self.edge_lin = qMLP([2 * dn, dn, dn])

    def forward(self, node_embd, edge_index, rbf_basis):

        xj = node_embd.index_select(0, edge_index[0])  # (|E|, dn)
        xi = node_embd.index_select(0, edge_index[1])  # (|E|, dn)

        ms = xj + xi  # (|E|, dn)
        rbf_w = self.rbf_lin(rbf_basis)  # (|E|, dn)
        ms = ms * rbf_w  # (|E|, dn)
        return ms
