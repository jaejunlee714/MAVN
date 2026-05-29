import torch

def RWSE(num_N, E, k):
    A = torch.zeros((num_N, num_N))
    for e in E:
        A[e[0], e[1]] = 1
    D_in = torch.sum(A, dim = 0)
    D_inv = torch.diag(torch.where(D_in == 0, 0, 1/torch.sum(A, dim = 0)))
    rw = A @ D_inv
    rwpe = []
    rw_k = torch.eye(num_N)
    for _ in range(k):
        rw_k = rw_k @ rw
        rwpe.append(torch.diagonal(rw_k))
    return torch.stack(rwpe, dim = 1)

def LapPE(num_N, E, k):
    A = torch.zeros((num_N, num_N))
    D = torch.zeros((num_N, num_N))
    for e in E:
        D[e[1], e[1]] += 1
        A[e[0], e[1]] = 1
    L = D - A
    eigval, eigvec = torch.linalg.eigh(L)
    idx = eigval.argsort()[:k]
    eigval = eigval[idx].unsqueeze(dim = 0)
    eigvec = torch.real(eigvec[:, idx])
    eigvec = torch.nn.functional.normalize(eigvec, dim = 1)
    if num_N < k:
        eigvec = torch.nn.functional.pad(eigvec, (0, k - num_N), value = 0)
        eigval = torch.nn.functional.pad(eigval, (0, k - num_N), value = 0)
    return torch.stack([eigvec, eigval.repeat(num_N, 1)], dim = 2)
