# Diffusion Model with Rician-Gaussian priors for Robust MRI Image Synthesis
![Image](https://github.com/user-attachments/assets/27c71f14-a97b-4f25-968f-6de151a68007)
This project introduces a Rician-Gaussian Diffusion Model for denoising and synthesizing medical MRI images, particularly tailored for noise-aware anomaly detection and segmentation tasks.
## How to use
### Train
```python
python3 diffusion_training.py ARGS_NUM
```
##### Example
```python
python3 diffusion_training.py args38
```

### Evaluation
```python
python3 evaluation.py ARGS_NUM
```
##### Example
```python
python3 evaluation.py args38
```
