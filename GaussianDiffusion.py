# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
import random

import matplotlib.pyplot as plt
import numpy as np

from importlib import import_module
from helpers import *

from pytorch_msssim import ssim
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn

import os

import torch.nn.functional as F
import torchvision.transforms as T


def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas


def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)


def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL Divergence between two gaussians

    :param mean1:
    :param logvar1:
    :param mean2:
    :param logvar2:
    :return: KL Divergence between N(mean1,logvar1^2) & N(mean2,logvar2^2))
    """
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))



def discretised_gaussian_log_likelihood(x, means, log_scales):
    """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.
        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
            )
    assert log_probs.shape == x.shape
    return log_probs

def add_rician_noise(x, sigma=0.1, rician_ratio=0.7, scale=0.002):
    """
    Rician + Gaussian 혼합 노이즈 추가 함수

    :param x: 입력 이미지 텐서 (B x C x H x W)
    :param sigma: 노이즈 강도
    :param rician_ratio: Rician 노이즈 비율 (0~1 사이), 나머지는 Gaussian
    :param scale: Rician 보정 스케일
    :return: 노이즈가 추가된 이미지
    """
    # Gaussian 노이즈
    noise_gauss = torch.randn_like(x) * sigma

    # Rician 노이즈
    noise_real = torch.randn_like(x) * sigma
    noise_imag = torch.randn_like(x) * sigma
    y = torch.sqrt((x + noise_real) ** 2 + noise_imag ** 2)

    bias = torch.sqrt(torch.clamp(x ** 2 + 2 * sigma ** 2, min=1e-8)) - x
    noise_rician = (y - bias) * scale

    # 혼합
    mixed_noise = rician_ratio * noise_rician + (1 - rician_ratio) * noise_gauss
    return mixed_noise

def simulate_rician_mri(x, sigma=0.1):
    """
    clean MRI magnitude image -> noisy MRI magnitude image
    """
    noise_real = torch.randn_like(x) * sigma
    noise_imag = torch.randn_like(x) * sigma
    y = torch.sqrt(torch.clamp((x + noise_real) ** 2 + noise_imag ** 2, min=1e-12))
    return y


def vst(x, sigma=0.1, eps=1e-8):
    """
    VST transform
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)

    while sigma.ndim < x.ndim:
        sigma = sigma.view(*sigma.shape, *([1] * (x.ndim - sigma.ndim)))

    return torch.sqrt(torch.clamp((x / (sigma + eps)) ** 2 + 3.0 / 8.0, min=0.0))


def inverse_vst(x, sigma=0.1, eps=1e-8):
    """
    approximate inverse VST
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)

    while sigma.ndim < x.ndim:
        sigma = sigma.view(*sigma.shape, *([1] * (x.ndim - sigma.ndim)))

    return (sigma + eps) * torch.sqrt(torch.clamp(x ** 2 - 3.0 / 8.0, min=0.0))


def normalize_vst(x, eps=1e-8):
    """
    normalize to [-1, 1]
    """
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    x_norm = 2.0 * (x - x_min) / (x_max - x_min + eps) - 1.0
    return x_norm, x_min, x_max


def denormalize_vst(x, x_min, x_max):
    """
    inverse normalization from [-1, 1]
    """
    return 0.5 * (x + 1.0) * (x_max - x_min) + x_min

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = [
            models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:4].eval(),
            models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[4:9].eval(),
        ]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        device = input.device  # ⚠️ 추가: GPU에서 연산할 수 있도록

        input = (input - self.mean.to(device)) / self.std.to(device)
        target = (target - self.mean.to(device)) / self.std.to(device)

        loss = 0.0
        for block in self.blocks:
            block = block.to(device)  # ⚠️ 각 블록을 GPU로 옮김
            input = block(input)
            target = block(target)
            loss += F.l1_loss(input, target)
        return loss

class GaussianDiffusionModel:
    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",  # l2,l1 hybrid
            loss_weight='none',  # prop t / uniform / None
            noise="rician",  # gauss / rician
            sigma=0.05,
            domain='rgdm'
            ):
        super().__init__()
        self.sigma = sigma
        self.domain = domain
        self.perceptual_loss = None if domain == "vst" else VGGPerceptualLoss()
        if noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)
        elif noise == "rician":
            self.noise_fn = lambda x, t: add_rician_noise(x, sigma=self.sigma)
        else:
            raise ValueError(f"Unsupported noise '{noise}'. Use 'gauss' or 'rician'.")

        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)

        self.loss_weight = loss_weight
        alphas = 1 - betas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],0.0)


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
                np.append(self.posterior_variance[1], self.posterior_variance[1:])
                )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )


    def sample_t_with_weights(self, b_size, device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)

    def generate_augmented_samples(self, model, x_0, num_augments=10, t_range=(100, 300), sigma_range=(0.05, 0.2)):
        aug_samples = []
        for _ in range(num_augments):
            sigma = np.random.uniform(*sigma_range)
            t_distance = random.choice(range(t_range[0], t_range[1] + 1, 50))

            self.sigma = sigma  # Rician 노이즈 강도 변경

            sample = self.forward_backward(
                model=model,
                x=x_0,
                t_distance=t_distance,
                see_whole_sequence=False,
                denoise_fn="rician",
            )
            aug_samples.append(sample)

        return torch.cat(aug_samples, dim=0)

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - pred_x_0) \
               / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(
                self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device
                )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """

        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))

        :param model:
        :param x_t:
        :param t:
        :return:
        """
        if estimate_noise == None:
            estimate_noise = model(x_t, t)

        # fixed model variance defined as \hat{\beta}_t - could add learned parameter
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(
                pred_x_0, x_t, t
                )
        return {
            "mean":         model_mean,
            "variance":     model_var,
            "log_variance": model_logvar,
            "pred_x_0":     pred_x_0,
            }

    def sample_p(self, model, x_t, t, denoise_fn="rician"):
        out = self.p_mean_variance(model, x_t, t)

        if self.domain == "vst":
            noise = torch.randn_like(x_t)
        else:
            if type(denoise_fn) == str:
                if denoise_fn == "gauss":
                    noise = torch.randn_like(x_t)
                elif denoise_fn == "rician":
                    noise = self.noise_fn(x_t, t).float()
                elif denoise_fn == "random":
                    noise = torch.randn_like(x_t)
                else:
                    noise = self.noise_fn(x_t, t).float()
            else:
                noise = self.noise_fn(x_t, t).float()

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def forward_backward(
            self, model, x, see_whole_sequence="half", t_distance=None, denoise_fn="rician",
            ):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps

        # VST domain에서는 입력 x를 raw image로 받아서 내부에서 변환
        if self.domain == "vst":
            x = x.clamp(0, 1)
            x = simulate_rician_mri(x, sigma=self.sigma)
            x = vst(x, sigma=self.sigma)
            x, x_vst_min, x_vst_max = normalize_vst(x)
            denoise_fn = "gauss"

        seq = [x.cpu().detach()]

        if see_whole_sequence == "whole":
            for t in range(int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])

                if self.domain == "vst":
                    noise = torch.randn_like(x)
                else:
                    noise = self.noise_fn(x, t_batch).float()

                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)

                seq.append(x.cpu().detach())
        else:
            t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])

            if self.domain == "vst":
                x = self.sample_q(x, t_tensor, torch.randn_like(x))
            else:
                x = self.sample_q(
                        x, t_tensor,
                        self.noise_fn(x, t_tensor).float()
                        )

            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model, x, t_batch, denoise_fn)
                x = out["sample"]
            if see_whole_sequence:
                seq.append(x.cpu().detach())

        if self.domain == "vst":
            x = denormalize_vst(x, x_vst_min, x_vst_max)
            x = inverse_vst(x, sigma=self.sigma).clamp(0, 1)

        return x.detach() if not see_whole_sequence else seq

    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )

            :param x_0:
            :param t:
            :param noise:
            :return:
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def sample_q_gradual(self, x_t, t, noise):
        """
        q (x_t | x_{t-1})
        :param x_t:
        :param t:
        :param noise:
        :return:
        """
        return (extract(self.sqrt_alphas, t, x_t.shape, x_t.device) * x_t +
                extract(self.sqrt_betas, t, x_t.shape, x_t.device) * noise)

    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        # find KL divergence at t
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, x_t, t)
        output = self.p_mean_variance(model, x_t, t, estimate_noise)
        kl = normal_kl(true_mean, true_log_var, output["mean"], output["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretised_gaussian_log_likelihood(
                x_0, output["mean"], log_scales=0.5 * output["log_variance"]
                )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x_0": output["pred_x_0"]}

    def calc_loss(self, model, x_0, t):
        x_0 = x_0.clamp(0, 1)

        # ==========================================================
        # VST branch
        # ==========================================================
        if self.domain == "vst":
            self.sigma = np.random.uniform(0.01, 0.10)

            # noisy input 생성 (diffusion input)
            x_noisy = simulate_rician_mri(x_0, sigma=self.sigma)

            # clean / noisy 둘 다 VST 변환
            x_vst_clean = vst(x_0, sigma=self.sigma)
            x_vst_noisy = vst(x_noisy, sigma=self.sigma)

            # normalize (같은 scale 사용)
            x_vst_noisy, x_min, x_max = normalize_vst(x_vst_noisy)
            x_vst_clean = 2.0 * (x_vst_clean - x_min) / (x_max - x_min + 1e-8) - 1.0

            # diffusion
            noise = torch.randn_like(x_vst_noisy)
            x_t = self.sample_q(x_vst_noisy, t, noise)
            estimate_noise = model(x_t, t)

            pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
            target_x_0 = x_vst_clean.clamp(-1, 1)

            if self.loss_type == "l1":
                recon_loss = F.l1_loss(pred_x_0, target_x_0)
            else:
                recon_loss = F.mse_loss(pred_x_0, target_x_0)

            perceptual_weight = 0.0
            perceptual = torch.tensor(0.0, device=x_0.device)

            total_loss = recon_loss
            loss = {"loss": total_loss, "perceptual": perceptual, "recon": recon_loss}
            return loss, x_t, estimate_noise

        # ==========================================================
        # original raw branch
        # ==========================================================
        self.sigma = np.random.uniform(0.01, 1.0)
        noise = add_rician_noise(x_0, sigma=self.sigma).float()
        x_t = self.sample_q(x_0, t, noise)
        estimate_noise = model(x_t, t)

        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(0, 1)
        target_x_0 = x_0.clamp(0, 1)

        brightness_weight = 1.0 - x_0.clamp(0, 1)
        brightness_weight = brightness_weight ** 2

        if self.loss_type == "l1":
            recon_loss = ((estimate_noise - noise).abs() * brightness_weight).mean()
        elif self.loss_type == "l2":
            recon_loss = ((estimate_noise - noise).square() * brightness_weight).mean()
        elif self.loss_type == "hybrid":
            out = self.calc_vlb_xt(model, x_0, x_t, t, estimate_noise)
            vlb = out["output"]
            recon_loss = vlb + ((estimate_noise - noise).square() * brightness_weight).mean()
        else:
            recon_loss = ((estimate_noise - noise).square() * brightness_weight).mean()

        perceptual_weight = 0.1
        perceptual = self.perceptual_loss(pred_x_0, target_x_0)

        total_loss = recon_loss + perceptual_weight * perceptual
        loss = {"loss": total_loss, "perceptual": perceptual, "recon": recon_loss}
        return loss, x_t, estimate_noise


    def p_loss(self, model, x_0, args):
        if self.loss_weight == "none":
            if args["train_start"]:
                t = torch.randint(
                        0, min(args["sample_distance"], self.num_timesteps), (x_0.shape[0],),
                        device=x_0.device
                        )
            else:
                t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            weights = 1
        else:
            t, weights = self.sample_t_with_weights(x_0.shape[0], x_0.device)

        loss, x_t, eps_t = self.calc_loss(model, x_0, t)
        loss = ((loss["loss"] * weights).mean(), (loss, x_t, eps_t))
        return loss

    def prior_vlb(self, x_0, args):
        t = torch.tensor([self.num_timesteps - 1] * args["Batch_Size"], device=x_0.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
        kl_prior = normal_kl(
                mean1=qt_mean, logvar1=qt_log_variance, mean2=torch.tensor(0.0, device=x_0.device),
                logvar2=torch.tensor(0.0, device=x_0.device)
                )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_total_vlb(self, x_0, model, args):
        vb = []
        x_0_mse = []
        mse = []
        for t in reversed(list(range(self.num_timesteps))):
            t_batch = torch.tensor([t] * args["Batch_Size"], device=x_0.device)
            noise = self.noise_fn(x_0, t_batch).float()
            x_t = self.sample_q(x_0=x_0, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self.calc_vlb_xt(
                        model,
                        x_0=x_0,
                        x_t=x_t,
                        t=t_batch,
                        )
            vb.append(out["output"])
            x_0_mse.append(mean_flat((out["pred_x_0"] - x_0) ** 2))
            eps = self.predict_eps_from_x_0(x_t, t_batch, out["pred_x_0"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        x_0_mse = torch.stack(x_0_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_vlb = self.prior_vlb(x_0, args)
        total_vlb = vb.sum(dim=1) + prior_vlb
        return {
            "total_vlb": total_vlb,
            "prior_vlb": prior_vlb,
            "vb":        vb,
            "x_0_mse":   x_0_mse,
            "mse":       mse,
            }
