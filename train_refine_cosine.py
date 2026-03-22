#!/usr/bin/env python
# coding=utf-8
"""
Train refinement network with frozen SD (ControlNet+UNet+VAE) and gating.

Inputs come from jsonl entries with fields:
  - conditioning_image: path to input image
  - image: path to GT image
  - prior: path to prior npy

Pipeline:
  cond -> diffusion (ControlNet -> gate -> UNet -> VAE decode) -> prelim (pixel)
  hf  <- wavelet high-frequency of cond
  prior <- prior npy to 1xHxW
  refine: NAFNet(prelim+hf+prior+cond) -> refined
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import lpips
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from basicsr.models.archs.NAFNet_arch import NAFNet
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, AutoTokenizer

from diffusion.controlnetvae import ControlNetVAEModel
from diffusion.pipeline_onestep import OneStepPipeline
from wavelet_color_fix import wavelet_decomposition


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train refinement network with frozen SD.")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory containing jsonl files.")
    parser.add_argument(
        "--multiple_datasets",
        type=str,
        nargs="+",
        required=True,
        help="List of jsonl filenames for training.",
    )
    parser.add_argument(
        "--multiple_datasets_probabilities",
        type=float,
        nargs="+",
        required=True,
        help="Sampling probabilities for each jsonl.",
    )
    parser.add_argument("--output_dir", type=str, default="refine_diff_outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--resize_scale", type=float, default=1.1)
    parser.add_argument("--disable_augment", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--l1_weight", type=float, default=0.5)
    parser.add_argument("--lpips_weight", type=float, default=0.25)
    parser.add_argument("--grad_weight", type=float, default=0.25)
    parser.add_argument("--nafnet_width", type=int, default=64)
    parser.add_argument("--nafnet_middle_blk_num", type=int, default=1)
    parser.add_argument(
        "--nafnet_enc_blk_nums",
        type=int,
        nargs="+",
        default=[1, 1, 1, 28],
        help="Encoder block numbers for NAFNet, e.g. --nafnet_enc_blk_nums 1 1 1 28",
    )
    parser.add_argument(
        "--nafnet_dec_blk_nums",
        type=int,
        nargs="+",
        default=[1, 1, 1, 1],
        help="Decoder block numbers for NAFNet, e.g. --nafnet_dec_blk_nums 1 1 1 1",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        "--model_id",
        type=str,
        default="./HF_CACHE/weights",
        help="Base model id/path for VAE, tokenizer and text encoder.",
    )
    parser.add_argument("--controlnet_dir", type=str, required=True)
    parser.add_argument("--unet_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="remove glass reflection")
    parser.add_argument("--beta", type=float, default=0.25)
    return parser.parse_args()


def load_jsonl(path: str) -> List[dict]:
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _normalize_to_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = (1, 2, 3)
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def compute_hf_image(image: torch.Tensor) -> torch.Tensor:
    high_freq, _ = wavelet_decomposition(image)
    hf = _normalize_to_01(high_freq)
    return hf


def compute_hf_mag(image: torch.Tensor) -> torch.Tensor:
    high_freq, _ = wavelet_decomposition(image)
    hf_mag = high_freq.abs().mean(dim=1, keepdim=True)
    mean = hf_mag.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    hf_mag = (hf_mag / mean).clamp(0.0, 1.0)
    return hf_mag


def load_prior_tensor(prior_path: str) -> torch.Tensor:
    prior = np.load(prior_path)
    if prior.ndim > 2:
        prior = prior.squeeze()
    prior = np.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
    prior_img = Image.fromarray((prior * 255.0).astype(np.uint8), mode="L")
    return TF.to_tensor(prior_img)  # [1,H,W]


class JsonlDataset(Dataset):
    def __init__(self, entries: List[dict], resolution: int, resize_scale: float, disable_augment: bool):
        self.entries = entries
        self.resolution = resolution
        self.resize_scale = resize_scale
        self.disable_augment = disable_augment

    def __len__(self) -> int:
        return len(self.entries)

    def _resize(self, img: Image.Image, size: int) -> Image.Image:
        return img.resize((size, size), Image.Resampling.BILINEAR)

    def __getitem__(self, idx: int) -> dict:
        item = self.entries[idx]
        cond_path = item["conditioning_image"]
        gt_path = item["image"]
        prior_path = item["prior"]

        cond = Image.open(cond_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        prior = load_prior_tensor(prior_path)
        prior = TF.to_pil_image(prior)

        if self.disable_augment:
            cond = self._resize(cond, self.resolution)
            gt = self._resize(gt, self.resolution)
            prior = self._resize(prior, self.resolution)
            if prior.size != cond.size:
                prior = prior.resize(cond.size, Image.Resampling.LANCZOS)
        else:
            resize_size = int(self.resolution * self.resize_scale)
            cond = self._resize(cond, resize_size)
            gt = self._resize(gt, resize_size)
            prior = self._resize(prior, self.resolution)
            if prior.size != cond.size:
                prior = prior.resize(cond.size, Image.Resampling.LANCZOS)

            i, j = torch.randint(0, resize_size - self.resolution + 1, (2,)).tolist()
            cond = TF.crop(cond, i, j, self.resolution, self.resolution)
            gt = TF.crop(gt, i, j, self.resolution, self.resolution)
            prior = TF.crop(prior, i, j, self.resolution, self.resolution)

            if random.random() < 0.5:
                cond = TF.hflip(cond)
                gt = TF.hflip(gt)
                prior = TF.hflip(prior)
            if random.random() < 0.5:
                cond = TF.vflip(cond)
                gt = TF.vflip(gt)
                prior = TF.vflip(prior)

        cond = TF.to_tensor(cond)  # [0,1]
        gt = TF.to_tensor(gt)      # [0,1]
        prior = TF.to_tensor(prior)  # [1,H,W]

        if prior.shape[-2:] != cond.shape[-2:]:
            raise ValueError(
                f"Prior size {prior.shape[-2:]} does not match cond size {cond.shape[-2:]}."
            )

        return {"cond": cond, "gt": gt, "prior": prior}


class FuseDataset(Dataset):
    def __init__(self, datasets: List[Dataset], probabilities: List[float]):
        self.datasets = datasets
        self.cum_probs = np.cumsum(probabilities)

    def __len__(self) -> int:
        return max(len(d) for d in self.datasets)

    def __getitem__(self, idx: int) -> dict:
        r = random.random()
        dataset_idx = int(np.searchsorted(self.cum_probs, r))
        dataset = self.datasets[dataset_idx]
        return dataset[idx % len(dataset)]


def load_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> OneStepPipeline:
    controlnet = ControlNetVAEModel.from_pretrained(args.controlnet_dir, torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.unet_dir, torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=dtype
    ).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )

    pipe = OneStepPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        scheduler=None,
        feature_extractor=None,
        t_start=0,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    for p in controlnet.parameters():
        p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    controlnet.eval()
    unet.eval()
    vae.eval()
    text_encoder.eval()
    return pipe


def infer_diff_prelim(
    pipeline: OneStepPipeline,
    image_tensor: torch.Tensor,
    prior_tensor: torch.Tensor,
    prompt: str,
    beta: float,
) -> torch.Tensor:
    device = pipeline._execution_device
    dtype = pipeline.dtype

    if pipeline.empty_text_embedding is None:
        text_inputs = pipeline.tokenizer(
            "",
            padding="do_not_pad",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        pipeline.empty_text_embedding = pipeline.text_encoder(text_input_ids)[0]

    if pipeline.prompt_embeds is None or pipeline.prompt != prompt:
        pipeline.prompt = prompt
        pipeline.prompt_embeds = None

    if pipeline.prompt_embeds is None:
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            pipeline.prompt,
            device,
            1,
            False,
            None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        pipeline.prompt_embeds = prompt_embeds
        pipeline.negative_prompt_embeds = negative_prompt_embeds

    image, padding, original_resolution = pipeline.image_processor.preprocess(
        image_tensor, pipeline.default_processing_resolution, "bilinear", device, dtype
    )
    if pipeline.prompt_embeds.shape[0] != image.shape[0]:
        pipeline.prompt_embeds = pipeline.prompt_embeds.repeat(image.shape[0], 1, 1)
    image_latent, pred_latent = pipeline.prepare_latents(image, None, None, 1, 1)

    prior = prior_tensor.to(device=device, dtype=dtype)
    prior = F.interpolate(prior, size=image.shape[-2:], mode="bilinear", align_corners=False)
    hf_mag = compute_hf_mag(image)

    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
        image_latent.detach(),
        pipeline.t_start,
        encoder_hidden_states=pipeline.prompt_embeds,
        conditioning_scale=1.0,
        guess_mode=False,
        return_dict=False,
    )

    gated_down = []
    prior_l = prior.clamp(0.0, 1.0)
    hf_l = hf_mag.clamp(0.0, 1.0)
    for res in down_block_res_samples:
        prior_res = F.interpolate(prior_l, size=res.shape[-2:], mode="area").clamp(0.0, 1.0)
        hf_res = F.interpolate(hf_l, size=res.shape[-2:], mode="area").clamp(0.0, 1.0)
        gate = 1.0 + beta * prior_res * hf_res
        gate = gate.clamp(1.0, 1.0 + beta)
        gated_down.append(res * gate)
    down_block_res_samples = gated_down

    prior_mid = F.interpolate(prior_l, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
    hf_mid = F.interpolate(hf_l, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
    gate_mid = 1.0 + beta * prior_mid * hf_mid
    gate_mid = gate_mid.clamp(1.0, 1.0 + beta)
    mid_block_res_sample = mid_block_res_sample * gate_mid

    latent_x_t = pipeline.unet(
        pred_latent,
        pipeline.t_start,
        encoder_hidden_states=pipeline.prompt_embeds,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
        return_dict=False,
    )[0]

    prediction = pipeline.decode_prediction(latent_x_t)
    prediction = pipeline.image_processor.unpad_image(prediction, padding)
    prediction = pipeline.image_processor.resize_antialias(
        prediction, original_resolution, "bilinear", is_aa=False
    )
    return prediction


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - gt))


def normalize_tensor_for_lpips(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 2.0) - 1.0


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    device = pred.device
    dtype = pred.dtype
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype)
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)

    channels = pred.shape[1]
    kx = kx.repeat(channels, 1, 1, 1)
    ky = ky.repeat(channels, 1, 1, 1)

    pred_dx = F.conv2d(pred, kx, padding=1, groups=channels)
    pred_dy = F.conv2d(pred, ky, padding=1, groups=channels)
    gt_dx = F.conv2d(gt, kx, padding=1, groups=channels)
    gt_dy = F.conv2d(gt, ky, padding=1, groups=channels)

    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)


def main() -> None:
    args = parse_args()
    if len(args.nafnet_enc_blk_nums) != len(args.nafnet_dec_blk_nums):
        raise ValueError("nafnet_enc_blk_nums and nafnet_dec_blk_nums must have the same length.")
    if args.nafnet_middle_blk_num < 0:
        raise ValueError("nafnet_middle_blk_num must be >= 0.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        tracker_config = dict(vars(args))
        tracker_config["nafnet_enc_blk_nums"] = ",".join(map(str, args.nafnet_enc_blk_nums))
        tracker_config["nafnet_dec_blk_nums"] = ",".join(map(str, args.nafnet_dec_blk_nums))
        tracker_config.pop("multiple_datasets", None)
        tracker_config.pop("multiple_datasets_probabilities", None)
        accelerator.init_trackers("refine_diff", config=tracker_config)

    if len(args.multiple_datasets) != len(args.multiple_datasets_probabilities):
        raise ValueError("multiple_datasets and multiple_datasets_probabilities must have same length.")

    probabilities = np.array(args.multiple_datasets_probabilities, dtype=np.float32)
    probabilities = probabilities / probabilities.sum()

    datasets = []
    for name in args.multiple_datasets:
        jsonl_path = os.path.join(args.train_data_dir, name)
        entries = load_jsonl(jsonl_path)
        datasets.append(
            JsonlDataset(entries, args.resolution, args.resize_scale, args.disable_augment)
        )

    train_dataset = FuseDataset(datasets, probabilities.tolist())
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = accelerator.device
    dtype = torch.float32

    pipeline = load_pipeline(args, device, dtype)

    in_ch = 10
    refine_net = NAFNet(
        img_channel=in_ch,
        width=args.nafnet_width,
        middle_blk_num=args.nafnet_middle_blk_num,
        enc_blk_nums=args.nafnet_enc_blk_nums,
        dec_blk_nums=args.nafnet_dec_blk_nums,
    ).to(device)
    refine_net.train()

    refine_head = torch.nn.Conv2d(in_ch, 3, kernel_size=1, bias=True).to(device)
    refine_head.train()

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        list(refine_net.parameters()) + list(refine_head.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def get_cosine_scheduler(optimizer, max_steps: int, warmup_steps: int, min_lr: float):
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0 for cosine scheduler.")

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return (min_lr / args.learning_rate) + (1.0 - min_lr / args.learning_rate) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    refine_net, refine_head, optimizer, train_dataloader = accelerator.prepare(
        refine_net, refine_head, optimizer, train_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps
    if max_train_steps is None:
        max_train_steps = args.epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_scheduler(
        optimizer,
        max_steps=max_train_steps,
        warmup_steps=args.lr_warmup_steps,
        min_lr=args.min_learning_rate,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.epochs}")
    logger.info(f"  Max train steps = {max_train_steps}")

    progress_bar = tqdm(
        range(0, max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(refine_net):
                cond = batch["cond"].to(device)
                gt = batch["gt"].to(device)
                prior = batch["prior"].to(device)

                with torch.no_grad():
                    prelim_pred = infer_diff_prelim(
                        pipeline,
                        cond,
                        prior,
                        args.prompt,
                        args.beta,
                    )
                    prelim = ((prelim_pred + 1.0) / 2.0).clamp(0.0, 1.0)
                    hf = compute_hf_image(cond)

                x = torch.cat([prelim, hf, prior, cond], dim=1)
                feat = refine_net(x)
                refined = refine_head(feat).clamp(0.0, 1.0)

                l1_val = l1_loss(refined, gt)
                lpips_val = lpips_model(
                    normalize_tensor_for_lpips(refined),
                    normalize_tensor_for_lpips(gt),
                ).mean()
                grad_val = gradient_loss(refined, gt)
                loss = (
                    args.l1_weight * l1_val
                    + args.lpips_weight * lpips_val
                    + args.grad_weight * grad_val
                )

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                lr_scheduler.step()
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    accelerator.save_state(ckpt_dir)

                    if args.checkpoints_total_limit is not None:
                        checkpoints = [
                            d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")
                        ]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) > args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                            for removing in checkpoints[:num_to_remove]:
                                removing_path = os.path.join(args.output_dir, removing)
                                try:
                                    import shutil
                                    shutil.rmtree(removing_path)
                                except OSError:
                                    pass

            logs = {
                "loss/total": loss.detach().item(),
                "loss/l1": l1_val.detach().item(),
                "loss/lpips": lpips_val.detach().item(),
                "loss/grad": grad_val.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(loss=logs["loss/total"])
            accelerator.log(
                {
                    "loss/total": logs["loss/total"],
                    "loss/l1": logs["loss/l1"],
                    "loss/lpips": logs["loss/lpips"],
                    "loss/grad": logs["loss/grad"],
                    "lr": logs["lr"],
                },
                step=global_step,
            )

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_net_path = os.path.join(args.output_dir, "nafnet_refine_final.pth")
        final_head_path = os.path.join(args.output_dir, "nafnet_refine_head_final.pth")
        torch.save(accelerator.unwrap_model(refine_net).state_dict(), final_net_path)
        torch.save(accelerator.unwrap_model(refine_head).state_dict(), final_head_path)


if __name__ == "__main__":
    main()
