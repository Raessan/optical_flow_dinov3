# Lightweight head for optical flow using DINOv3 as backbone

This repository provides a lightweight optical flow head designed to run on top of Meta’s [DINOv3](https://github.com/facebookresearch/dinov3) backbone. The model implements local correlation and depthwise separable convolutions to estimate dense motion fields from DINOv3 features with a compact convex upsampling module. It has been trained using the [Flying chairs dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) and the [Flying things 3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html).

This head is part of the [dinov3_ros](https://github.com/Raessan/dinov3_ros) project, where it enables real-time semantic segmentation in ROS 2 by reusing backbone features across multiple perception tasks.


## Table of Contents

1. [Installation](#installation)
2. [Model and loss function](#model_loss)
3. [Usage](#usage)
4. [Integration with dinov3_ros](#integration_dinov3_ros)
5. [Demo](#demo)
6. [License](#license)
7. [References](#references)


## Installation

We recommend using a fresh `conda` environment to keep dependencies isolated. DINOv3 requires Python 3.11, so we set that explicitly.

```
conda create -n optical_flow_dinov3 python=3.11
conda activate optical_flow_dinov3
git clone --recurse-submodules https://github.com/Raessan/optical_flow_dinov3
cd optical_flow_dinov3
pip install -e .
```

The only package that has to be installed separately is PyTorch, due to its dependence with the CUDA version. For example:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129 
```

Finally, we provide weights for the lightweight heads developed by us in the `weights` folder, but the DINOv3 backbone weights should be requested and obtained from their [repo](https://github.com/facebookresearch/dinov3). Its default placement is in `dinov3_weights` folder. The presented head has been trained using the `vits16plus` model from DINOv3 as a backbone.

Training also requires the the [Flying chairs dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) and the [Flying things 3D dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) to be installed on disk.

## Model and loss function

This repository implements a lightweight optical flow head that can be attached to the [DINOv3](https://github.com/facebookresearch/dinov3) backbone (or any ViT producing a single spatial feature map). The architecture emphasizes efficiency and accuracy in dense motion estimation through local correlation, separable convolutions, and learned upsampling.

### Model architecture

#### Local Correlation Encoder

- Computes a local cost volume between two feature maps using a configurable search radius.

- Captures fine motion cues by correlating each feature with its spatial neighborhood.

#### Feature Fusion and FlowRegression

- Projects DINOv3 features into a lower-dimensional space and fuses them together with the cost volume and absolute feature difference.

- Processes the fused representation through depthwise separable convolution and squeeze-and-excitation (SE) blocks to enhance efficiency and channel attention.

- Predicts an initial optical flow field at the feature resolution.

#### Refinement and Convex Upsampling

- Optionally refines the coarse flow through an additional residual block operating at the feature scale.

- Uses a RAFT-style convex upsampler to interpolate the low-resolution flow into a high-resolution, edge-aware motion field while preserving fine spatial details.

### Loss functions

Training minimizes a robust end-point error (EPE) loss between the predicted and ground-truth flow fields. Invalid pixels and extreme displacements are masked out, ensuring that the loss focuses on valid motion regions. The total loss is defined as:

$$
L_{total} = \| (F_{pred} - F_{gt}) \odot M \|_{1}
$$

## Usage

There are three main folders and files that the user should use:

`config/config.py`: This file allows the user to configure the model, loss, training step or inference step. The parameters are described in the file.

`train/train_optical_flow.ipynb`: Jupyter notebook for training the optical flow algorithm. It can load and/or save checkpoints depending on the configuration in `config.py`.

`inference/inference.py`: Script for running inference with a trained model on new images.

Additionally, the repository includes a `src` folder that contains the backend components: dataset utilities, backbone/head model definitions, and helper scripts. In particular:

- `common.py`: general-purpose functions that can be reused across different task-specific heads.
- `utils.py`: utilities tailored specifically for optical flow (e.g., generate an image representing the flow).

The optical flow was trained for a total of 35 epochs: first for 15 epochs with the [Flying chairs dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) dataset, with a learning rate of 1e-4 using data augmentation, then 15 epochs with the [Flying things 3D dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) with a learning rate of 1e-4 using data augmentation. Finally, 5 epochs with the [Flying things 3D dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) with a reduced learning rate of 1e-5 without augmentation. The final weights have been placed in the `weights` folder.

Our main objective was not to surpass state-of-the-art models, but to train a head with solid results that enables collaboration and contributes to building a more refined [dinov3_ros](https://github.com/Raessan/dinov3_ros). This effort is particularly important because Meta has not released lightweight task-specific heads. For this reason, we welcome contributions — whether it’s improving this optical flow head, adding new features, or experimenting with alternative model architectures. Feel free to open an issue or submit a pull request! See the [Integration with dinov3_ros](#integration-dinov3_ros) section to be compliant with the [dinov3_ros](https://github.com/Raessan/dinov3_ros) project. 

## Integration with [dinov3_ros](https://github.com/Raessan/dinov3_ros)

This repository is designed to be easily integrated into [dinov3_ros](https://github.com/Raessan/dinov3_ros). To enable plug-and-play usage, the following files must be exported from the `src` folder to the `dinov3_toolkit/head_optical_flow` folder in [dinov3_ros](https://github.com/Raessan/dinov3_ros):

- `model_head.py`: defines the detection head architecture.
- `utils.py`: task-specific utilites for optical flow.

Additionally, we provide our chosen weights in `weights/model.pth`.

Any modification or extension of this repository should maintain these files and remain self-contained, so that the head can be directly plugged into [dinov3_ros](https://github.com/Raessan/dinov3_ros) without additional dependencies.

## Demo

<img src="assets/gif_semantic_segmentation.gif" height="800">

## License
- Code in this repo: Apache-2.0.
- DINOv3 submodule: licensed separately by Meta (see its LICENSE).
- We don't distribute DINO weights. Follow upstream instructions to obtain them.

## References

- [Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, Piotr Bojanowski (2025). Dinov3. *arXiv preprint arXiv:2508.10104.*](https://github.com/facebookresearch/dinov3)

- [Philipp Fischer, Alexey Dosovitskiy, Eddy Ilg, Philip Häusser, Caner Hazırbaş, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, Thomas Brox (2015). Flownet: Learning optical flow with convolutional networks. *IEEE International Conference on Computer Vision (ICCV)*](https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/)

- [Nikolaus Mayer, Eddy Ilg, Philip Häusser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, Thomas Brox (2016). A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. *Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4040-4048)*.](https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16/)

- [Teed Zachary, Jia Deng (2020). Raft: Recurrent all-pairs field transforms for optical flow. *European conference on computer vision*. Cham: Springer International Publishing](https://arxiv.org/abs/2003.12039)

- [Escarabajal, Rafael J. (2025). dinov3_ros](https://github.com/Raessan/dinov3_ros)