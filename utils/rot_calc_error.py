import numpy as np
from tqdm import tqdm

def _flip(rot):
    x = np.diag([1, 1, -1]).astype(rot.dtype)
    return np.matmul(x, rot)


def get_ref_matrix(r1, r2, i, flip=False):
    if flip:
        return np.matmul(r2[i].T, _flip(r1[i]))
    else:
        return np.matmul(r2[i].T, r1[i])


def align_rot(r1, r2, i, flip=False):
    if flip:
        ref_mat = get_ref_matrix(r1, r2, i, flip=True)
        return np.matmul(_flip(r2), ref_mat), ref_mat
    else:
        ref_mat = get_ref_matrix(r1, r2, i, flip=False)
        return np.matmul(r2, ref_mat), ref_mat 


def get_ang_dist(dist):
    dist = np.clip(dist, 0, 8)
    ang_dist = np.arcsin(np.sqrt(dist / 8))*2 / np.pi * 180
    return ang_dist


def get_ang_dist_from_cosine(cosine_dist):
    cosine_dist = np.clip(cosine_dist, -1, 1)
    ang_dist_rad = np.arccos(cosine_dist)
    ang_dist = ang_dist_rad / np.pi * 180
    return ang_dist


def get_rotation_accuracy(pred_rotmat, gt_rotmat, flip, ref_mat=None, error_type='mean', n_samples=None):
    if ref_mat is None:
        mean = []
        median = []
        fullrot_l2_dists = []
        ref_mats = []
        viewdir_cosine_dists = []
        if n_samples == None:
            samples = np.arange(pred_rotmat.shape[0])
        else:
            samples = np.arange(n_samples)
        for i in tqdm(samples):
            pred_rotmat_aligned, ref_mat = align_rot(gt_rotmat, pred_rotmat, i, flip=flip)
            l2_dists = np.sum((gt_rotmat - pred_rotmat_aligned) ** 2, axis=(1, 2))
            viewdir_cosine_dist = (gt_rotmat[:, 2] * pred_rotmat_aligned[:, 2]).sum(-1)
            mean.append(np.mean(l2_dists))
            median.append(np.median(l2_dists))
            fullrot_l2_dists.append(l2_dists)
            viewdir_cosine_dists.append(viewdir_cosine_dist)
            ref_mats.append(ref_mat)
        min_mean_error = np.min(mean)
        min_median_error = np.min(median)
        min_mean_idx = np.argmin(mean)
        min_median_idx = np.argmin(median)
        if error_type == 'mean':
            return min_mean_error, ref_mats[min_mean_idx], get_ang_dist(fullrot_l2_dists[min_mean_idx]), get_ang_dist_from_cosine(viewdir_cosine_dists[min_mean_idx])
        elif error_type == 'median':
            return min_median_error, ref_mats[min_median_idx], get_ang_dist(fullrot_l2_dists[min_median_idx]), get_ang_dist_from_cosine(viewdir_cosine_dists[min_median_idx])
        else:
            raise NotImplementedError
            
    else:
        if flip:
            pred_rotmat_aligned = np.matmul(_flip(pred_rotmat), ref_mat)
        else:
            pred_rotmat_aligned = np.matmul(pred_rotmat, ref_mat)
        return pred_rotmat_aligned
        # dot_prod = (pred_rotmat_aligned[:, 2]*gt_rotmat[:, 2]).sum(-1)
        # viewdir_ang_dist = get_ang_dist_from_cosine(dot_prod)

        # dists = np.sum((gt_rotmat - pred_rotmat_aligned) ** 2, axis=(1, 2))
        # full_ang_dist = get_ang_dist(dists)
        # return full_ang_dist, viewdir_ang_dist


def compute_rot_error_single(pred_rotmats, gt_rotmats, n_samples, error_type='mean'):
    orig_min_error, orig_ref_mat, fullrot_dists, viewdir_dists = get_rotation_accuracy(pred_rotmats, gt_rotmats, 
                                                                    flip=False, ref_mat=None, 
                                                                    error_type=error_type, n_samples=n_samples)
    flip_min_error, flip_ref_mat, flip_fullrot_dists, flip_viewdir_dists = get_rotation_accuracy(pred_rotmats, gt_rotmats, 
                                                                         flip=True, ref_mat=None, 
                                                                         error_type=error_type, n_samples=n_samples)
    if flip_min_error < orig_min_error:
        out_dict = {'full_rot_error': flip_fullrot_dists,
                'viewdir_error': flip_viewdir_dists,
                'flip': True,
                'ref_mat': flip_ref_mat}
    else:
        out_dict = {'full_rot_error': fullrot_dists,
                'viewdir_error': viewdir_dists,
                'flip': False,
                'ref_mat': orig_ref_mat}
    return out_dict
    
def align_all_heads(pred_rotmats, gt_rotmats, ref_mat, flip):
    full_ang_dists_heads = []
    viewdir_ang_dists_heads = []
    for head in range(pred_rotmats.shape[1]):
        if flip:
            head_i_aligned_pred_rotmats = np.matmul(_flip(pred_rotmats[:, head]), ref_mat)
        else:
            head_i_aligned_pred_rotmats = np.matmul(pred_rotmats[:, head], ref_mat)
        l2_dists = np.sum((gt_rotmats - head_i_aligned_pred_rotmats) ** 2, axis=(1, 2))
        full_ang_dists = get_ang_dist(l2_dists)
        viewdir_cosine_dist = (gt_rotmats[:, 2] * head_i_aligned_pred_rotmats[:, 2]).sum(-1)
        viewdir_ang_dists = get_ang_dist_from_cosine(viewdir_cosine_dist)
        full_ang_dists_heads.append(full_ang_dists)
        viewdir_ang_dists_heads.append(viewdir_ang_dists)
    full_ang_dists_heads = np.stack(full_ang_dists_heads, axis=1)
    viewdir_ang_dists_heads = np.stack(viewdir_ang_dists_heads, axis=1)
    return full_ang_dists_heads, viewdir_ang_dists_heads