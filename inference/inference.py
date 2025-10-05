import torch
from torch import nn
import numpy as np
from src.model_head import LiteFlowHead
from src.model_backbone import DinoBackbone
from src.common import image_to_tensor
from src.utils import flow_to_image
import config.config as cfg
import cv2
import sys
import time
import matplotlib.pyplot as plt

IMG_SIZE = cfg.IMG_SIZE
PATCH_SIZE = cfg.PATCH_SIZE
IMG_MEAN = np.array(cfg.IMG_MEAN, dtype=np.float32)[:, None, None]
IMG_STD = np.array(cfg.IMG_STD, dtype=np.float32)[:, None, None]

DINOV3_DIR = cfg.DINOV3_DIR
DINO_MODEL = cfg.DINO_MODEL
DINO_WEIGHTS = cfg.DINO_WEIGHTS
MODEL_TO_NUM_LAYERS = cfg.MODEL_TO_NUM_LAYERS
MODEL_TO_EMBED_DIM = cfg.MODEL_TO_EMBED_DIM

PROJ_CHANNELS = cfg.PROJ_CHANNELS 
RADIUS = cfg.RADIUS
FUSION_CHANNELS = cfg.FUSION_CHANNELS
FUSION_LAYERS = cfg.FUSION_LAYERS
CONVEX_UP = cfg.CONVEX_UP
REFINEMENT_LAYERS = cfg.REFINEMENT_LAYERS

MODEL_PATH_INFERENCE = cfg.MODEL_PATH_INFERENCE
IMG_INFERENCE_PATH_1 = cfg.IMG_INFERENCE_PATH_1
IMG_INFERENCE_PATH_2 = cfg.IMG_INFERENCE_PATH_2


device = "cuda" if torch.cuda.is_available() else "cpu"

n_layers_dino = MODEL_TO_NUM_LAYERS[DINO_MODEL]
dino_model = torch.hub.load(
        repo_or_dir=DINOV3_DIR,
        model=DINO_MODEL,
        source="local",
        weights=DINO_WEIGHTS
)
dino_backbone = DinoBackbone(dino_model, n_layers_dino).to(device)

embed_dim = MODEL_TO_EMBED_DIM[DINO_MODEL]

model_head = LiteFlowHead(out_size = IMG_SIZE,
        in_channels = embed_dim,
        proj_channels = PROJ_CHANNELS,
        radius = RADIUS,
        fusion_channels = FUSION_CHANNELS,
        fusion_layers = FUSION_LAYERS,
        convex_up = CONVEX_UP,
        refinement_layers = REFINEMENT_LAYERS).to(device)

model_head.load_state_dict(torch.load(MODEL_PATH_INFERENCE))

# Process image 1
image1 = cv2.imread(IMG_INFERENCE_PATH_1)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Resize image
image1 = cv2.resize(image1, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
image1_tensor = image_to_tensor(image1, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)

# Process image 2
image2 = cv2.imread(IMG_INFERENCE_PATH_2)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Resize image
image2 = cv2.resize(image2, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
image2_tensor = image_to_tensor(image2, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)

# Inference
dino_backbone.eval()
model_head.eval()

with torch.no_grad():
    feat1 = dino_backbone(image1_tensor)
    feat2 = dino_backbone(image2_tensor)
    init = time.time()
    n_inference = 1000
    for i in range(n_inference):
        flow = model_head(feat1, feat2)
    end = time.time()
    print("time per sample: ", (end-init)*1000/n_inference)

flow_pred_plot = flow_to_image(flow.squeeze().permute(1,2,0).cpu().numpy())

# Put the images in a list and plot them
images = [image1, image2, flow_pred_plot]
titles = ["Image 1", "Image 2", "Flow Pred"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
