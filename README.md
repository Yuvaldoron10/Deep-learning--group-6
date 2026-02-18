# Multi-Focus Image Fusion (Extended IFCNN)
Deep Learning Course Project, Ben-Gurion University  
Authors: Yuval Doron, Shaked Zur Shaday

## Description
This project improves the IFCNN image fusion model for the Multi-Focus Image Fusion task — combining multiple images of the same scene into a single fully-focused image.

We extend the original model from 2 input images to 4 input images and systematically evaluate architectural and training modifications.

## Results
The final model improves perceptual quality (SSIM, VIFF) while maintaining similar PSNR compared to the original IFCNN baseline.

## Project Structure

### `models/`
All network architectures:
- `IFCNN model.py` — original baseline
- `IFCNN with adjustments.py`
- `improved_model.py` — final proposed model
- `model2_conv1.py` — conv1 fine-tuning experiment
- `model3_activation.py` — activation experiments
- `model4_residual.py` — early residual blocks
- `model5_deeper.py` — post-fusion refinement

### `train and val/`
Training scripts:
- `train+val original model.py`
- `train+val improved model.py`

### `test/`
Evaluation scripts:
- `test_original.py`
- `test_improved_model.py`

### `results_test/`
Example qualitative outputs:
- `results_original/`
- `results_improved/`

### `weights/`
- `IFCNN-MAX.pth` — baseline weights from the original paper




