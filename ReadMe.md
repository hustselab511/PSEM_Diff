# PSEM-Diff

PSEM-Diff is an innovative framework designed for high-fidelity conversion of Ballistocardiogram (BCG) signals to Electrocardiogram (ECG) signals.  The method effectively enhances cross-modal transformations through pre-encoding, multiband diffusion, and dynamic noise adjustment.

## Key Features
- **Extraction of ECG Physiological Event Temporal Semantics**: Improves cross-modal transformations by capturing and encoding meaningful ECG event semantics.
- **Multi-band Diffusion**: Utilizes separate diffusion models for different frequency bands to enhance cross-modal semantic mapping and denoising.
- **Dynamic Noise Adjustment**: Adapts noise levels dynamically based on the characteristics of different signal segments, improving reconstruction accuracy.
- **Support for Downstream Tasks**: Enables applications such as identity recognition and atrial fibrillation (AF) detection.

## Applications
- **Identity Recognition**: Achieves high accuracy in distinguishing individual physiological patterns.
- **Atrial Fibrillation Detection**: Facilitates early and non-invasive screening for cardiovascular abnormalities.

## Environment
Before you can start using the MuSFId model, you need to install the following dependencies:
- Python 3.7+
- TensorFlow 2.0+
- NumPy 1.16+
- matplotlib==3.7.1
- numpy==1.24.2
- pandas==2.0.3
- scikit-learn==1.3.2
- scipy==1.10.1
- seaborn==0.13.0
- sklearn==0.0
- sympy==1.11.1
- tensorboard==2.12.0
- torch @ http://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp38-cp38-linux_x86_64.whl
- torchvision @ http://download.pytorch.org/whl/cu118/torchvision-0.15.1%2Bcu118-cp38-cp38-linux_x86_64.whl
- tqdm @ file:///tmp/build/80754af9/tqdm_1625563689033/work



## Quick start
1. Train stage1：
   ```bash
   cd stage1
   python unet_train.py
   
2. Training the stage2：
   ```bash
   cd stage1
   python reconstruct_train.py

3. Training the stage3：
   ```bash
   cd stage1
   python diffusion_train.py