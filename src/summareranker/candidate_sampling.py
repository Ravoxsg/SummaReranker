import numpy as np
import torch



def candidate_subsampling(mode, ids, masks, scores, labels, args):
    selected_idx = list(range(len(ids)))
    # remove duplicates
    if torch.sum(mode) > 0 and args.filter_out_duplicates:
        idx = unique_idx(scores)
        selected_idx = [selected_idx[idx[k]] for k in range(len(idx))]
        ids = ids[idx]
        masks = masks[idx]
        scores = scores[:, idx]
        labels = labels[:, idx]
    # only select a few positive and a few negative candidates
    if torch.sum(mode) > 0 and args.prune_candidates:
        idx_to_keep = prune_idx(scores, args)
        idx_to_keep = idx_to_keep[:args.max_n_candidates]
        selected_idx = [selected_idx[idx_to_keep[k]] for k in range(len(idx_to_keep))]
        ids = ids[idx_to_keep]
        masks = masks[idx_to_keep]
        scores = scores[:, idx_to_keep]
        labels = labels[:, idx_to_keep]

    return selected_idx, ids, masks, scores, labels


def unique_idx(t):
    if len(t.shape) == 2:
        reduced_t = t.sum(axis = 0)
    else:
        reduced_t = t
    idx = []
    items = []
    for i in range(len(reduced_t)):
        if not(reduced_t[i].item() in items):
            items.append(reduced_t[i].item())
            idx.append(i)
    idx = np.array(idx)
    p = np.random.permutation(len(idx))
    idx = idx[p]
    idx = list(idx)

    return idx


def prune_idx(scores, args):
    s = scores.detach().cpu().numpy()
    # take top + bottom ones with regards to the sum 
    if len(s.shape) == 2:
        reduced_s = np.sum(s, 0)
    else:
        reduced_s = s
    sort_idx = np.argsort(reduced_s)[::-1]
    if args.sampling_strat == "bottom":
        neg = list(sort_idx[-args.n_negatives:])
    elif args.sampling_strat == "random":
        p = np.random.permutation(len(sort_idx) - args.n_positives)
        neg = list(sort_idx[args.n_positives:][p][:args.n_negatives])
    idx_to_keep = list(sort_idx)[:args.n_positives] + neg

    idx_to_keep = np.array(idx_to_keep)
    p = np.random.permutation(len(idx_to_keep))
    idx_to_keep = idx_to_keep[p]
    idx_to_keep = list(idx_to_keep)

    return idx_to_keep



