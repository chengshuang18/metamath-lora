import torch
import functools
import torch.distributed as dist
from accelerate.utils import DeepSpeedEngineWrapper

def get_module(root_module, module_name):
    """Retrieve a submodule from a root module based on the module's name."""
    attrs = module_name.split('.')
    return functools.reduce(getattr, attrs, root_module)

def apply_importance_mask(name, module, importance_mask):
    """Apply the importance mask to the gradients of a module's weights."""
    if hasattr(module, 'weight') and module.weight.numel() != 0:
        assert module.weight.grad is not None, f"{module} has no grad"
        module.weight.grad *= importance_mask.unsqueeze(dim=-1).to(module.weight.device)

def compute_importance_mask(activation, ini_threshold, n_cluster):
    """Compute the importance mask based on the provided method."""
    device = activation[0].device
    hidden_dim = activation.shape[-1]
    dist.all_reduce(activation, op=dist.ReduceOp.AVG)
    if n_cluster != None:
        activation_chunks = activation.chunk(n_cluster)
        activation = torch.stack([chunk.sum() for chunk in activation_chunks])
        threshold = torch.quantile(activation, ini_threshold)
        importance_mask = (activation >= threshold).float().to(device)
        assert hidden_dim % n_cluster ==0, "hidden_dim must be divisible by n_cluster."
        importance_mask = importance_mask.repeat_interleave(hidden_dim // n_cluster)
    else:
        threshold = torch.quantile(activation, ini_threshold)
        importance_mask = (activation >= threshold).float().to(device)
    return importance_mask

def mask_gradient(accelerator, model, activations, args):
    ori_model = accelerator.unwrap_model(model)
    for name, activation in activations.items():
        if "lora_A" in name:
            importance_mask = compute_importance_mask(activation, args.threshold, None)
            module = get_module(ori_model, name)
        else:
            n_cluster = None if args.no_cluster else args.n_cluster
            importance_mask = compute_importance_mask(activation, args.threshold, n_cluster)
            module = get_module(ori_model, name + '.lora_B.default')

        apply_importance_mask(name, module, importance_mask)
            
