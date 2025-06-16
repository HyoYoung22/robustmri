import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_curve
import torchvision.utils as vutils
import numpy as np
import os

from helpers import gridify_output


def heatmap(real: torch.Tensor, recon: torch.Tensor, mask, filename, save=True):
    mse = ((recon - real).square() * 2) - 1
    mse_threshold = mse > 0
    mse_threshold = (mse_threshold.float() * 2) - 1
    if save:
        output = torch.cat((real, recon.reshape(1, *recon.shape), mse, mse_threshold, mask))
        plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
        plt.axis('off')
        plt.savefig(filename)
        plt.clf()


def dice_coeff(real: torch.Tensor, recon: torch.Tensor, real_mask: torch.Tensor, smooth=1e-6, mse=None):
    if mse is None:
        mse = (real - recon).square()
        mse = (mse > 0.5).float()
    intersection = torch.sum(mse * real_mask, dim=[1, 2, 3])
    union = torch.sum(mse, dim=[1, 2, 3]) + torch.sum(real_mask, dim=[1, 2, 3])
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
    return dice


def PSNR(recon, real):
    se = (real - recon).square()
    mse = torch.mean(se, dim=list(range(len(real.shape))))
    psnr = 20 * torch.log10(torch.max(real) / torch.sqrt(mse))
    return psnr.detach().cpu().numpy()


def SSIM(real, recon):
    real_np = real.detach().cpu().numpy()
    recon_np = recon.detach().cpu().numpy()
    return ssim(real_np, recon_np, channel_axis=2, data_range=1.0)


def IoU(real, recon):
    real = real.cpu().numpy()
    recon = recon.cpu().numpy()
    intersection = np.logical_and(real, recon)
    union = np.logical_or(real, recon)
    return np.sum(intersection) / (np.sum(union) + 1e-8)


def precision(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FP = ((real_mask == 1) & (recon_mask == 0))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FP)).float() + 1e-6)


def recall(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FN = ((real_mask == 0) & (recon_mask == 1))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FN)).float() + 1e-6)


def FPR(real_mask, recon_mask):
    FP = ((real_mask == 1) & (recon_mask == 0))
    TN = ((real_mask == 0) & (recon_mask == 0))
    return torch.sum(FP).float() / ((torch.sum(FP) + torch.sum(TN)).float() + 1e-6)


def ROC_AUC(real_mask, square_error):
    if isinstance(real_mask, torch.Tensor):
        return roc_curve(real_mask.detach().cpu().numpy().flatten(), square_error.detach().cpu().numpy().flatten())
    else:
        return roc_curve(real_mask.flatten(), square_error.flatten())


def AUC_score(fpr, tpr):
    return auc(fpr, tpr)


def testing(testing_dataset_loader, diffusion, args, ema, model):
    ema.eval()
    model.eval()

    base_save_dir = f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/inferenceimage(0.2)/'
    input_dir = os.path.join(base_save_dir, 'input')
    output_dir = os.path.join(base_save_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    t_step = 500
    print(f"\n Sampling at step t={t_step} on full test set...")
    img_idx = 0

    for batch in testing_dataset_loader:
        if args["dataset"] in ["cifar", "carpet"]:
            x = batch[0].to(device)
        else:
            x = batch["image"].to(device)

        recon = diffusion.forward_backward(ema, x, see_whole_sequence=None, t_distance=t_step)[-1]

        for i in range(x.size(0)):
            input_img = x[i].unsqueeze(0)
            output_img = recon[i].unsqueeze(0)

            base_input_name = f'input-{t_step}-img{img_idx}'
            base_output_name = f'output-{t_step}-img{img_idx}'

            input_npy_path = os.path.join(input_dir, base_input_name + '.npy')
            input_png_path = os.path.join(input_dir, base_input_name + '.png')
            output_npy_path = os.path.join(output_dir, base_output_name + '.npy')
            output_png_path = os.path.join(output_dir, base_output_name + '.png')

            #np.save(input_npy_path, input_img.squeeze(0).detach().cpu().numpy())
            #np.save(output_npy_path, output_img.squeeze(0).detach().cpu().numpy())

            vutils.save_image(input_img, input_png_path, normalize=True)
            vutils.save_image(output_img, output_png_path, normalize=True)

            img_idx += 1

    test_iters = 40
    vlb = []
    for epoch in range(test_iters // args["Batch_Size"] + 5):
        data = next(testing_dataset_loader)
        if args["dataset"] == "cifar":
            x = data[0].to(device)
        else:
            x = data["image"].to(device)

        vlb_terms = diffusion.calc_total_vlb(x, model, args)
        vlb.append(vlb_terms)

    psnr = []
    for epoch in range(test_iters // args["Batch_Size"] + 5):
        data = next(testing_dataset_loader)
        if args["dataset"] == "cifar":
            x = data[0].to(device)
        else:
            x = data["image"].to(device)

        out = diffusion.forward_backward(ema, x, see_whole_sequence=None, t_distance=500 // 2)
        psnr.append(PSNR(out, x))

    print(
        f"Test set total VLB: {np.mean([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])} "
        f"+- {np.std([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])}")
    print(
        f"Test set prior VLB: {np.mean([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])} "
        f"+- {np.std([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])}")
    print(
        f"Test set vb @ t=200: {np.mean([i['vb'][0][199].cpu().item() for i in vlb])} "
        f"+- {np.std([i['vb'][0][199].cpu().item() for i in vlb])}")
    print(
        f"Test set x_0_mse @ t=200: {np.mean([i['x_0_mse'][0][199].cpu().item() for i in vlb])} "
        f"+- {np.std([i['x_0_mse'][0][199].cpu().item() for i in vlb])}")
    print(
        f"Test set mse @ t=200: {np.mean([i['mse'][0][199].cpu().item() for i in vlb])} "
        f"+- {np.std([i['mse'][0][199].cpu().item() for i in vlb])}")
    print(f"Test set PSNR: {np.mean(psnr)} +- {np.std(psnr)}")


def main():
    args, output = load_parameters(device)
    print(f"args{args['arg_num']}")

    in_channels = 3 if args["dataset"].lower() == "cifar" else 1
    unet = UNetModel(
        args['img_size'][0], args['base_channels'],
        channel_mults=args['channel_mults'], in_channels=in_channels
    )
    ema = UNetModel(
        args['img_size'][0], args['base_channels'],
        channel_mults=args['channel_mults'], in_channels=in_channels
    )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
        args['img_size'], betas,
        loss_weight=args['loss_weight'],
        loss_type=args['loss-type'],
        noise=args["noise_fn"]
    )

    ema.load_state_dict(output["ema"])
    ema.to(device)
    ema.eval()

    unet.load_state_dict(output["model_state_dict"])
    unet.to(device)
    unet.eval()

    _, testing_dataset = dataset.init_datasets("./", args)
    testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    testing(testing_dataset_loader, diff, args, ema, unet)


if __name__ == '__main__':
    import dataset
    from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
    from UNet import UNetModel
    from detection import load_parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
