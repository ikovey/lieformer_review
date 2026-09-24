"""Opt-in, bounded attention summaries. Never retain edge tensors or graphs."""

import torch


class AttentionDiagnostics:
    def __init__(self):
        self.enabled = False
        self.context = {}
        self.rows = []

    def capture(self, block, logits, alpha, dst, num_nodes):
        if not self.enabled:
            return
        with torch.no_grad():
            a = alpha.detach().float()
            counts = torch.bincount(dst, minlength=num_nodes)
            valid = counts > 0
            entropy = a.new_zeros((num_nodes, a.shape[1]))
            entropy.index_add_(0, dst, -a * a.clamp_min(1e-30).log())
            gram = a.T @ a
            norms = gram.diag().sqrt()
            cosine = gram / (norms[:, None] * norms[None, :]).clamp_min(1e-30)
            row = dict(
                self.context,
                block=block,
                edges=len(dst),
                targets=int(valid.sum()),
                entropy=entropy[valid].mean(0).cpu().tolist() if valid.any() else None,
                head_cosine=cosine.cpu().tolist() if len(dst) else None,
                logit_gradient_rms=None,
            )
            self.rows.append(row)
        if logits.requires_grad:

            def record_gradient(grad):
                with torch.no_grad():
                    row["logit_gradient_rms"] = (
                        grad.detach().float().square().mean(0).sqrt().cpu().tolist() if grad.shape[0] else None
                    )

            logits.register_hook(record_gradient)

    def take(self):
        rows, self.rows = self.rows, []
        return rows
