import torch
from typing import Callable, Optional


def contrastive_loss_sc_hnm(
        z1: torch.Tensor,
        z2: torch.Tensor,
        init1: torch.Tensor,
        init2: torch.Tensor,
        temperature: float = 0.07,
        top_k: int = 5,
        delta: float = 0.9,
        log_fn: Optional[Callable] = None
) -> tuple[torch.Tensor, int]:
    """
    Vectorised InfoNCE with SC-HNM filtering.
    Returns loss (scalar) and num_false_negatives_filtered (int).
    """
    B = z1.size(0)
    device = z1.device

    # Final space for the loss, initial space for filtering.
    sim_final = torch.matmul(z1, z2.T) / temperature
    sim_init = torch.matmul(init1, init2.T)

    # Build the "do-not-touch" mask.
    diag_mask = torch.eye(B, dtype=torch.bool, device=device)
    # High similarity in the initial space? Probably a False Negative.
    fn_mask = sim_init > delta
    invalid_mask = diag_mask | fn_mask

    # Mine for the hardest negatives.
    # First, nuke the invalid pairs so we don't pick them.
    sim_clean = sim_final.masked_fill(invalid_mask, float('-inf'))

    _, topk_idx = torch.topk(sim_clean, top_k, dim=1)
    hard_mask = torch.zeros_like(sim_clean, dtype=torch.bool)
    hard_mask.scatter_(1, topk_idx, True)

    # The final, legit negatives for the loss calculation.
    neg_mask = hard_mask & (~invalid_mask)

    # --- Vectorized InfoNCE Magic ---
    pos_logits = torch.diag(sim_final)
    logits_neg = sim_final.masked_fill(~neg_mask, float('-inf'))

    # This compact logsumexp is elegant.
    denom = torch.logsumexp(
        torch.cat([pos_logits.unsqueeze(1), logits_neg], dim=1), dim=1
    )
    loss = -(pos_logits - denom).mean()

    # --- Stats for debugging/logging ---
    num_fn_filtered = (fn_mask & (~diag_mask)).sum().item()

    if log_fn is not None:
        log_fn(num_fn=num_fn_filtered)

    return loss, num_fn_filtered