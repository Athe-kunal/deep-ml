import torch

def moe_load_balancing_loss(gate_logits, num_experts: int, alpha: float = 0.01) -> float:
    """
    Compute the load balancing auxiliary loss for a Mixture of Experts layer.
    
    Args:
        gate_logits: torch.Tensor of shape (num_tokens, num_experts), raw gating scores
        num_experts: int, number of experts
        alpha: float, scaling coefficient for the loss
    
    Returns:
        float: load balancing loss rounded to 4 decimal places
    """
    gate_softmax = torch.softmax(gate_logits,dim=-1) #(num_tokens, num_experts)
    experts_indices = torch.argmax(gate_softmax,dim=-1) #(num_tokens, num_experts)
    f = torch.nn.functional.one_hot(experts_indices, num_experts).float().mean(dim=0)
    pi = gate_softmax.mean(dim=0)
    return alpha * num_experts * torch.sum(f * pi, dim=-1).item()

