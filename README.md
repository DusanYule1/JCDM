# JCDM: Jointly Conditioned Diffusion Model for Multi-View Pose-Guided Person Image Synthesis

This repository contains the official implementation of JCDM, a diffusion-based framework that synthesizes high-fidelity person images from multiple reference views and a target pose. It addresses the limitations of single-view methods by leveraging multi-view priors to generate consistent and realistic images, even with significant pose changes.

JCDM consists of two key components:

1. **Appearance Prior Module (APM):** Infers a holistic, identity-preserving prior from incomplete reference images.
2. **Joint Conditional Injection (JCI):** Fuses multi-view cues and injects them as shared conditioning into the denoising backbone to align identity, color, and texture across different poses.

## Training

The training process is divided into two main stages.

### Stage 1: Train the Appearance Prior Module (APM)

This stage trains the APM to predict a complete semantic embedding from masked multi-view inputs.

```
sh run_train_stage1.sh
```

### Stage 2: Train the Denoising U-Net

This stage trains the denoising U-Net with the frozen APM to generate the final image. This is a two-part process. Run the following scripts in order:

```
# First part of Stage 2 training
sh run_train_stage2_stage1.sh

# Second part of Stage 2 training
sh run_train_stage2_stage2.sh
```

## Inference

To generate images using a trained model, follow these steps.

### Stage 1 Inference

Run inference with the trained Appearance Prior Module (APM).

```
sh test_stage1.sh
```

### Stage 2 Inference

There are two options for Stage 2 inference: with or without the motion module.

#### Without Motion Module

To run inference without the motion module for static image generation:

```
python test_stage2_no_motion.py
```

#### With Motion Module

To run inference with the motion module for generating video frames or sequences:

```
python test_stage2_use_motion.py
```

# JCDM：用于多视图姿态引导的虚拟人图像合成的联合条件扩散模型

本代码库包含 JCDM 的官方实现，这是一个基于扩散的框架，可以从多个参考视图和目标姿态合成高保真度的虚拟人图像。它通过利用多视图先验来生成一致且逼真的图像，即使在姿态变化很大的情况下，也解决了单视图方法的局限性。

JCDM 由两个关键组件组成：

1. **外观先验模块 (APM):** 从不完整的参考图像中推断出整体的、保持身份的先验。
2. **联合条件注入 (JCI):** 融合多视图线索，并将其作为共享条件注入到去噪主干中，以在不同姿态之间对齐身份、颜色和纹理。

## 训练

训练过程分为两个主要阶段。

### 第一阶段：训练外观先验模块 (APM)

该阶段训练 APM 从带掩码的多视图输入中预测完整语义嵌入。

```
sh run_train_stage1.sh
```

### 第二阶段：训练去噪 U-Net

该阶段使用冻结的 APM 训练去噪 U-Net 以生成最终图像。这是一个分为两部分的过程，请依次运行以下脚本：

```
# 第二阶段训练的第一部分
sh run_train_stage2_stage1.sh

# 第二阶段训练的第二部分
sh run_train_stage2_stage2.sh
```

## 推理

要使用经过训练的模型生成图像，请按照以下步骤操作。

### 第一阶段推理

使用训练好的外观先验模块 (APM) 进行推理。

```
sh test_stage1.sh
```

### 第二阶段推理

第二阶段推理有两个选项：带或不带运动模块。

#### 不带运动模块

要在没有运动模块的情况下运行推理以生成静态图像：

```
python test_stage2_no_motion.py
```

#### 带运动模块

要使用运动模块运行推理以生成视频帧或序列：

```
python test_stage2_use_motion.py
```