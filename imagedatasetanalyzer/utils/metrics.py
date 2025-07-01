import os

import torch
import lpips
import piq 
import cv2
import numpy as np
from tqdm import tqdm

def _load_images_from_folder(path, resize_to=(384, 384), normalize='01', device='cpu'):

    img_paths = sorted([
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(('.png', '.jpg'))
    ])

    imgs = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        if resize_to:
            img = cv2.resize(img, resize_to)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        if normalize == '-11':
            img = img * 2 - 1  # For LPIPS
        # else: default is [0, 1]

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        imgs.append(img_tensor.to(device))

    return imgs


@torch.no_grad()
def compute_pairwise_similarity(imgs, metric_fn, batch_size=16, desc="Computing similarity"):
    N = len(imgs)
    dists = []

    for i in tqdm(range(N), desc=desc):
        img_i = imgs[i]
        comparisons = imgs[i+1:]

        for j in range(0, len(comparisons), batch_size):
            batch = torch.cat(comparisons[j:j+batch_size], dim=0)
            img_i_batch = img_i.expand(batch.size(0), -1, -1, -1)

            dist_vals = metric_fn(img_i_batch, batch)
            dists.extend(dist_vals.tolist())

    dists = np.array(dists)
    return dists.mean(), dists.std(), dists


# === Specific metric wrappers ===

@torch.no_grad()
def compute_SSIM(path, device, resize_to=(512, 512), batch_size=16):
    imgs = _load_images_from_folder(path, resize_to=resize_to, normalize='01', device=device)

    def ssim_fn(x, y):
        return piq.ssim(x, y, data_range=1.0, reduction='none')

    return compute_pairwise_similarity(imgs, ssim_fn, batch_size=batch_size, desc="Computing SSIM")


@torch.no_grad()
def compute_LPIPS(path, device, resize_to=(384, 384), batch_size=16):
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()

    imgs = _load_images_from_folder(path, resize_to=resize_to, normalize='-11', device=device)

    def lpips_fn(x, y):
        return loss_fn(x, y)

    return compute_pairwise_similarity(imgs, lpips_fn, batch_size=batch_size, desc="Computing LPIPS")