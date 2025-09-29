# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2

import os
import math
import random
from glob import glob
import os.path as osp

from src.frame_utils import read_gen
from src.dataset_augmentor import FlowAugmentor
from src.common import image_to_tensor

def check_nan(name, arr):
    if isinstance(arr, np.ndarray):
        if np.isnan(arr).any():
            print(f"{name} has NaNs")
            return True
    elif torch.is_tensor(arr):
        if torch.isnan(arr).any():
            print(f"{name} has NaNs")
            return True
    return False

class FlowDataset(data.Dataset):
    def __init__(self, img_size, img_mean, img_std, aug_params=None):
        self.img_size = img_size
        self.img_mean = np.array(img_mean, dtype=np.float32)[:, None, None]
        self.img_std = np.array(img_std, dtype=np.float32)[:, None, None]
        self.prob_augment = aug_params.pop("prob_augment")

        self.augmentor = None
        if aug_params is not None:
            self.augmentor = FlowAugmentor(**aug_params)

        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        flow = read_gen(self.flow_list[index])

        # After loading flow
        if check_nan("flow initial", flow):
            print("Index NaN: ", index)

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        img1 = img1[..., :3]
        img2 = img2[..., :3]

        # Resize image and flow
        H, W = img1.shape[:2]
        Ht, Wt = self.img_size

        # images: bilinear
        img1 = cv2.resize(img1, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

        # flow: bilinear resample + component scaling
        flow = cv2.resize(flow, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        sx, sy = Wt / W, Ht / H
        flow[..., 0] *= sx
        flow[..., 1] *= sy
        check_nan("flow after resize", flow)

        # Apply augmentor
        if self.augmentor is not None and np.random.rand() < self.prob_augment:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        check_nan("flow after augmentor", flow)

        img1 = image_to_tensor(img1, self.img_mean, self.img_std)
        img2 = image_to_tensor(img2, self.img_mean, self.img_std)

        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        check_nan("flow tensor", flow)

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()
        
    def __len__(self):
        return len(self.image_list)
        


class FlyingChairs(FlowDataset):
    def __init__(self, img_size, img_mean, img_std, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(img_size, img_mean, img_std, aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('../chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='train' and xid==1) or (split=='val' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]

class FlyingThings3D(FlowDataset):
    def __init__(self, img_size, img_mean, img_std, aug_params=None, split='train', root='datasets/FlyingThings3D'):
        super(FlyingThings3D, self).__init__(img_size, img_mean, img_std, aug_params)

        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, 'image_clean')

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(image_root, cam)))
                flow_dirs = sorted(glob(osp.join(flow_root, cam, direction)))

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.flo')) )

                    # Map frame index -> image path (expects zero-padded numeric stems, e.g., 0000123.png)
                    idx2img = {}
                    for img_path in images:
                        stem = osp.splitext(osp.basename(img_path))[0]
                        try:
                            idx2img[int(stem)] = img_path
                        except ValueError:
                            # Skip non-numeric filenames
                            continue

                    # Build matched pairs
                    for flow_path in flows:
                        flow_stem = osp.splitext(osp.basename(flow_path))[0]
                        try:
                            k = int(flow_stem)
                        except ValueError:
                            continue

                        if direction == 'into_future':
                            i0, i1 = k, k + 1   # XXXXXXX.flo -> XXXXXXX.png, (XXXXXXX+1).png
                        else:  # 'into_past'
                            i0, i1 = k, k - 1   # XXXXXXX.flo -> XXXXXXX.png, (XXXXXXX-1).png

                        if i0 in idx2img and i1 in idx2img:
                            self.image_list += [[idx2img[i0], idx2img[i1]]]
                            self.flow_list  += [flow_path]

def fetch_dataset(dataset_name, dataset_locations, mode, img_size, img_mean, img_std, prob_augment=0.0):
    """ Create the data loader for the corresponding trainign set """

    if dataset_name == 'flying_chairs':
        aug_params = {'crop_size': img_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True, 'prob_augment': prob_augment}
        train_dataset = FlyingChairs(img_size, img_mean, img_std, aug_params, split=mode, root = dataset_locations[dataset_name])

    elif dataset_name == 'flying_things_3d':
        aug_params = {'crop_size': img_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True, 'prob_augment': prob_augment}
        train_dataset = FlyingThings3D(img_size, img_mean, img_std, aug_params, split=mode, root = dataset_locations[dataset_name])

    return train_dataset

if __name__ == "__main__":
    # Dataset variables
    DATASET_NAME = "flying_things_3d" # Type of dataset
    DATASET_LOCATIONS = {
        'flying_chairs': "/home/rafa/deep_learning/datasets/FlyingChairs_release/data",
        "flying_things_3d": "/home/rafa/deep_learning/datasets/FlyingThings3D_subset"
    }
    IMG_SIZE = (640, 640)
    IMG_MEAN = [0.485, 0.456, 0.406] # Mean of the image that the backbone (e.g. ResNet) expects
    IMG_STD = [0.229, 0.224, 0.225] # Std of the image that the backbone (e.g. ResNet) expects
    PROB_AUGMENT = 0.0
    MODE = "train"

    dataset = fetch_dataset(DATASET_NAME, DATASET_LOCATIONS, MODE, IMG_SIZE, IMG_MEAN, IMG_STD, PROB_AUGMENT)
    from common import tensor_to_image
    from utils import flow_to_image
    sample = dataset[5000]
    im1 = tensor_to_image(sample[0], IMG_MEAN, IMG_STD)
    im2 = tensor_to_image(sample[1], IMG_MEAN, IMG_STD)
    flow = sample[2]
    valid = sample[3]
    # map flow to rgb image
    flow_image = flow_to_image(flow.permute(1,2,0).cpu().numpy())

    import matplotlib.pyplot as plt
    from check_flow import photometric_check_v2

    err, cov, img2_w = photometric_check_v2(torch.from_numpy(im1).permute(2,0,1), torch.from_numpy(im2).permute(2,0,1), flow, valid_mask=valid.unsqueeze(0))
    print(f"Photometric L1: {err:.4f}, coverage: {cov*100:.1f}%")

    # img1
    plt.figure()
    plt.imshow(im1)
    plt.title("img1 orig")
    plt.axis('off')
    plt.tight_layout()

    # img2 warped
    plt.figure()
    plt.imshow(img2_w.squeeze(0).permute(1,2,0).cpu().numpy())
    plt.title("img2 warped")
    plt.axis('off')
    plt.tight_layout()

    # img2
    plt.figure()
    plt.imshow(im2)
    plt.title("img2")
    plt.axis('off')
    plt.tight_layout()

    # flow color
    plt.figure()
    plt.imshow(flow_image)
    plt.title("flow (color-coded)")
    plt.axis('off')
    plt.tight_layout()

    plt.show()
