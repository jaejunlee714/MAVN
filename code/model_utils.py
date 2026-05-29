import torch
import torch.nn.functional as F

def GroupSoftmax(num_groups, groups, logits, weights = None):
    max_per_group = torch.zeros([num_groups]+list(logits.shape)[1:]).cuda().index_reduce(dim = 0, index  = groups, source = logits.detach().clone(), reduce = "amax", include_self = False)
    if weights is None:
        weights = torch.ones_like(logits)
    exps = weights * torch.exp(logits - max_per_group[groups])
    total_per_group = torch.zeros_like(max_per_group).index_add(dim = 0, index = groups, source = exps)
    return exps / (torch.index_select(total_per_group, 0, groups) + 1e-6)

def GroupLogSoftmax(num_groups, groups, logits):
    max_per_group = torch.zeros([num_groups]+list(logits.shape)[1:]).cuda().index_reduce(dim = 0, index  = groups, source = logits.detach().clone(), reduce = "amax", include_self = False)
    normalized_logits = logits - max_per_group[groups]
    total_per_group = torch.zeros_like(max_per_group).index_add(dim = 0, index = groups, source = torch.exp(normalized_logits))
    return  normalized_logits - torch.log(torch.index_select(total_per_group, 0, groups))

def GroupLogSumExp(num_groups, groups, logits):
    max_per_group = torch.zeros([num_groups]+list(logits.shape)[1:]).cuda().index_reduce(dim = 0, index  = groups, source = logits.detach().clone(), reduce = "amax", include_self = False)
    total_per_group = torch.zeros_like(max_per_group).index_add(dim = 0, index = groups, source = torch.exp(logits - max_per_group[groups]))
    return  max_per_group + torch.log(total_per_group)

def GroupNormalize(cnt_groups, groups, vectors, mean_weights, eps = 1e-6):
    num_groups = len(cnt_groups)
    group_mean = torch.zeros((num_groups, vectors.size()[1])).cuda().index_add(dim = 0, index = groups, source = vectors) / cnt_groups.reshape(-1,1)
    group_var = torch.zeros((num_groups, vectors.size()[1])).cuda().index_add(dim = 0, index = groups, source = (vectors - mean_weights * torch.index_select(group_mean, 0, groups))**2) / cnt_groups.reshape(-1,1)
    return (vectors - mean_weights * torch.index_select(group_mean, 0, groups)) / torch.sqrt(torch.index_select(group_var, 0, groups) + eps)

def GumbelSoftmax(num_groups, groups, logits, tau: float, weights = None):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau
    return GroupSoftmax(num_groups, groups, gumbels, weights = weights)

def GumbelSigmoid(logits, tau: float):
    gumbels1 = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels2 = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )

    values = (logits + gumbels1 - gumbels2) / tau

    return F.sigmoid(values)