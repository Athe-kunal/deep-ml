import torch

def moe(x: torch.Tensor, We: torch.Tensor, Wg: torch.Tensor, n_experts: int, top_k: int) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    router_logits = x @ Wg #(n_batch, l_seq, n_experts)
    assert top_k <= n_experts
    experts = torch.topk(router_logits,top_k)
    # (n_batch, l_seq, top_k)
    expert_indices, expert_values = experts.indices, experts.values
    # (top_k, d_model, d_model)
    expert_weights = We[expert_indices,:,:]
    # x(n_batch, l_seq, d_model) @ We(top_k, d_model, d_model)
    # x -> (n_batch, l_seq, 1, 1, d_model) @ (top_k, d_model, d_model) -> (n_batch, l_seq, top_k, 1, d_model)
    return (torch.softmax(expert_values, dim=-1).unsqueeze(-1) * ((x.unsqueeze(-2).unsqueeze(-2) @ expert_weights ).squeeze(-2))).sum(dim=-2)