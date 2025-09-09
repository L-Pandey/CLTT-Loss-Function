import torch
import torch.nn.functional as F

''''
if dataset = [x1,x2,x3],[x2,x3,x4],[x3,x4,x5],[x4,x5,x6],.... where x{i} are frames arranged
in a temporal window, then out_1, out_2, and out_3 passed to the loss function are embeddings 
taken from a model. 

out_1 = [x1,x2,x3,x4] first frame from all the temporal windows
out_2 = [x2,x3,x4,x5] second frame from all the temporal windows
out_3 = [x3,x4,x5,x6] third frame from all the temporal windows
'''

def temporal_nt_xent_loss_debug(out_1, out_2, out_3, temperature=0.1, eps=1e-6):
    N, d = out_1.shape

    # Normalize embeddings
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    out_3 = F.normalize(out_3, dim=-1)

    # Stack windows: [N,3,d]
    windows = torch.stack([out_1, out_2, out_3], dim=1)
    reps = windows.view(-1, d)  # Flatten: [3*N, d]

    # Similarity matrix
    sim_matrix = torch.mm(reps, reps.t()) / temperature
    sim_matrix = torch.exp(sim_matrix)

    # Positive mask
    mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    for i in range(N):
        idx = torch.arange(i*3, i*3+3)
        for ii in idx:
            for jj in idx:
                mask[ii, jj] = True

    # Negative mask
    neg_mask = ~mask

    # Numerator: sum of positive similarities per anchor
    numerator = sim_matrix[mask].view(N*3, 3).sum(dim=1)  # [3*N] anchors
    # Denominator: sum of all positives+negatives per anchor
    denominator = sim_matrix * neg_mask.float()  # zeros out positives
    denominator = denominator.sum(dim=1) + numerator  # [3*N] anchors

    # Per-anchor loss
    loss_per_anchor = -torch.log((numerator + eps) / (denominator + eps))
    loss = loss_per_anchor.mean()

    # Debug prints (if needed)
    print("Stacked windows:\n", windows)
    print("Flattened embeddings:\n", reps)
    print("Similarity matrix:\n", sim_matrix)
    print("Positive mask:\n", mask)
    print("Negative mask:\n", neg_mask)
    print("Numerator per anchor:\n", numerator)
    print("Denominator per anchor:\n", denominator)
    print("Loss per anchor:\n", loss_per_anchor)
    print("Final loss:", loss)

    return loss