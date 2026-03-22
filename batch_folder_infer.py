"""
Run diffusion + refinement on a folder of blended images with matching prior .npy files.

Each blended image (jpg/png) must have a same-name .npy prior in prior_dir.
Results are saved to output_dir as <name>_diff.png and <name>_refine.png.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Tuple

import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from basicsr.models.archs.NAFNet_arch import NAFNet
from diffusion.controlnetvae import ControlNetVAEModel
from diffusion.pipeline_onestep import OneStepPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, AutoTokenizer

from wavelet_color_fix import wavelet_decomposition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference for diffusion + refinement."
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        choices=["folder", "jsonl"],
        required=True,
        help="Use explicit folder paths or jsonl-driven input.",
    )
    parser.add_argument("--blended_dir", type=str, default=None, help="Folder with blended images.")
    parser.add_argument("--prior_dir", type=str, default=None, help="Folder with prior .npy files.")
    parser.add_argument("--jsonl_dir", type=str, default=None, help="Directory containing jsonl files.")
    parser.add_argument(
        "--jsonl_files",
        type=str,
        nargs="+",
        default=None,
        help="Jsonl files with `conditioning_image` and `prior` fields.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to save results in folder mode.")
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="If max(h,w) exceeds this, scale the long side to max_size.",
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
    parser.add_argument("--refine_net_path", type=str, required=True)
    parser.add_argument("--refine_head_path", type=str, required=True)
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
    parser.add_argument("--prompt", type=str, default="remove glass reflection")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of CUDA devices to use. 0 means use all visible GPUs automatically.",
    )
    parser.add_argument(
        "--noimg",
        action="store_true",
        help="Do not save outputs, only benchmark inference time.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of initial images to skip for warmup timing stats.",
    )
    return parser.parse_args()


def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            files.append(os.path.join(folder, name))
    return sorted(files)


def load_jsonl_entries(path: str) -> List[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def build_tasks_from_jsonl(jsonl_dir: str, jsonl_files: List[str]) -> List[Tuple[str, str]]:
    tasks: List[Tuple[str, str]] = []
    for jsonl_name in jsonl_files:
        jsonl_path = jsonl_name if os.path.isabs(jsonl_name) else os.path.join(jsonl_dir, jsonl_name)
        for item in load_jsonl_entries(jsonl_path):
            img_path = item.get("conditioning_image")
            prior_path = item.get("prior")
            if img_path is None or prior_path is None:
                raise ValueError(f"Missing conditioning_image/prior in {jsonl_path}")
            tasks.append((img_path, prior_path))
    return tasks


def build_tasks_from_dirs(blended_dir: str, prior_dir: str) -> List[Tuple[str, str]]:
    tasks: List[Tuple[str, str]] = []
    for img_path in list_images(blended_dir):
        base = os.path.splitext(os.path.basename(img_path))[0]
        prior_path = os.path.join(prior_dir, f"{base}.npy")
        tasks.append((img_path, prior_path))
    return tasks


def get_output_paths(args: argparse.Namespace, img_path: str) -> Tuple[str, str]:
    stem = Path(img_path).stem + ".png"
    if args.input_mode == "folder":
        if args.output_dir is None:
            raise ValueError("--output_dir is required in folder mode.")
        diff_path = os.path.join(args.output_dir, f"{Path(img_path).stem}_diff.png")
        refine_path = os.path.join(args.output_dir, f"{Path(img_path).stem}_refine.png")
        return diff_path, refine_path

    parent_dir = Path(img_path).parent
    sibling_root = parent_dir.parent
    diff_path = str(sibling_root / "diff" / stem)
    refine_path = str(sibling_root / "refine" / stem)
    return diff_path, refine_path


def filter_pending_tasks(
    args: argparse.Namespace,
    tasks: List[Tuple[str, str]],
    noimg: bool,
) -> List[Tuple[str, str]]:
    if noimg:
        return tasks
    pending = []
    for img_path, prior_path in tasks:
        diff_path, refine_path = get_output_paths(args, img_path)
        if os.path.exists(diff_path) and os.path.exists(refine_path):
            continue
        pending.append((img_path, prior_path))
    return pending


def resize_long_side(img: Image.Image, max_size: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_size:
        return img
    scale = max_size / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


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


def load_refine_models(args: argparse.Namespace, device: torch.device) -> tuple[NAFNet, torch.nn.Module]:
    in_ch = 10
    refine_net = NAFNet(
        img_channel=in_ch,
        width=args.nafnet_width,
        middle_blk_num=args.nafnet_middle_blk_num,
        enc_blk_nums=args.nafnet_enc_blk_nums,
        dec_blk_nums=args.nafnet_dec_blk_nums,
    ).to(device)
    refine_head = torch.nn.Conv2d(in_ch, 3, kernel_size=1, bias=True).to(device)

    refine_net.load_state_dict(torch.load(args.refine_net_path, map_location="cpu"))
    refine_head.load_state_dict(torch.load(args.refine_head_path, map_location="cpu"))
    refine_net.eval()
    refine_head.eval()
    return refine_net, refine_head


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


def load_prior_tensor(prior_path: str, target_size: Tuple[int, int]) -> torch.Tensor:
    prior = np.load(prior_path)
    if prior.ndim > 2:
        prior = prior.squeeze()
    prior = np.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
    prior_img = Image.fromarray((prior * 255.0).astype(np.uint8), mode="L")
    if prior_img.size != target_size:
        prior_img = prior_img.resize(target_size, Image.Resampling.LANCZOS)
    prior_tensor = TF.to_tensor(prior_img).unsqueeze(0)  # [1,1,H,W]
    return prior_tensor


def infer_with_diff(
    pipeline: OneStepPipeline,
    image_tensor: torch.Tensor,
    prompt: str,
    prior_tensor: torch.Tensor,
    beta: float,
) -> np.ndarray:
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
    image_latent, pred_latent = pipeline.prepare_latents(image, None, None, 1, 1)

    prior = prior_tensor.to(device=device, dtype=dtype)
    prior = F.interpolate(prior, size=image.shape[-2:], mode="bilinear", align_corners=False)
    hf_mag = compute_hf_mag(image)

    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
        image_latent.detach(),
        pipeline.t_start,
        encoder_hidden_states=pipeline.prompt_embeds,
        controlnet_cond=image,
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
    prediction = pipeline.image_processor.pt_to_numpy(prediction)[0]
    return prediction


def refine_image(
    refine_net: NAFNet,
    refine_head: torch.nn.Module,
    cond_tensor: torch.Tensor,
    prelim_tensor: torch.Tensor,
    prior_tensor: torch.Tensor,
) -> torch.Tensor:
    hf = compute_hf_image(cond_tensor)
    x = torch.cat([prelim_tensor, hf, prior_tensor, cond_tensor], dim=1)
    feat = refine_net(x)
    refined = refine_head(feat).clamp(0.0, 1.0)
    return refined


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    tensor = (tensor * 255.0 + 0.5).to(torch.uint8)
    return tensor.permute(0, 2, 3, 1).numpy()[0]


def process_single_task(
    pipeline: OneStepPipeline,
    refine_net: NAFNet,
    refine_head: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    task: Tuple[str, str],
) -> bool:
    img_path, prior_path = task
    if not os.path.exists(prior_path):
        print(f"[Skip] missing prior: {prior_path}")
        return False

    blended = Image.open(img_path).convert("RGB")
    blended = resize_long_side(blended, args.max_size)
    blended_tensor = TF.to_tensor(blended).unsqueeze(0).to(device)

    w, h = blended.size
    prior_tensor = load_prior_tensor(prior_path, (w, h)).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        output = infer_with_diff(
            pipeline,
            blended_tensor,
            args.prompt,
            prior_tensor,
            args.beta,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    pred_tensor = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0)
    pred_tensor = ((pred_tensor + 1) / 2).clamp(0, 1).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        refined_tensor = refine_image(
            refine_net,
            refine_head,
            blended_tensor,
            pred_tensor,
            prior_tensor,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    if not args.noimg:
        diff_img = Image.fromarray(tensor_to_uint8(pred_tensor))
        refine_img = Image.fromarray(tensor_to_uint8(refined_tensor))
        diff_path, refine_path = get_output_paths(args, img_path)
        os.makedirs(os.path.dirname(diff_path), exist_ok=True)
        os.makedirs(os.path.dirname(refine_path), exist_ok=True)
        diff_img.save(diff_path)
        refine_img.save(refine_path)
    return True


def run_worker(args: argparse.Namespace, tasks: List[Tuple[str, str]], device_str: str, worker_idx: int) -> None:
    device = torch.device(device_str)
    dtype = torch.float32
    pipeline = load_pipeline(args, device, dtype)
    refine_net, refine_head = load_refine_models(args, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    warmup_start = None
    warmup_processed = 0
    processed = 0

    for task in tqdm(tasks, desc=f"Worker {worker_idx} ({device_str})", total=len(tasks)):
        success = process_single_task(pipeline, refine_net, refine_head, args, device, task)
        if not success:
            continue
        processed += 1
        if processed == args.warmup + 1:
            warmup_start = time.time()
        if processed > args.warmup and warmup_start is not None:
            warmup_processed += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    if processed > 0:
        avg = elapsed / processed
        print(f"[Worker {worker_idx}] Processed {processed} images in {elapsed:.2f}s (avg {avg:.4f}s/img).")
    if warmup_processed > 0 and warmup_start is not None:
        warmup_elapsed = time.time() - warmup_start
        warmup_avg = warmup_elapsed / warmup_processed
        print(
            f"[Worker {worker_idx}] Post-warmup {warmup_processed} images in {warmup_elapsed:.2f}s "
            f"(avg {warmup_avg:.4f}s/img, warmup={args.warmup})."
        )


def split_tasks_round_robin(tasks: List[Tuple[str, str]], num_parts: int) -> List[List[Tuple[str, str]]]:
    chunks: List[List[Tuple[str, str]]] = [[] for _ in range(num_parts)]
    for idx, task in enumerate(tasks):
        chunks[idx % num_parts].append(task)
    return chunks


def main() -> None:
    args = parse_args()
    if args.input_mode == "folder":
        if args.blended_dir is None or args.prior_dir is None or args.output_dir is None:
            raise ValueError("Folder mode requires --blended_dir, --prior_dir and --output_dir.")
    else:
        if args.jsonl_dir is None or not args.jsonl_files:
            raise ValueError("Jsonl mode requires --jsonl_dir and --jsonl_files.")
    if len(args.nafnet_enc_blk_nums) != len(args.nafnet_dec_blk_nums):
        raise ValueError("nafnet_enc_blk_nums and nafnet_dec_blk_nums must have the same length.")
    if args.nafnet_middle_blk_num < 0:
        raise ValueError("nafnet_middle_blk_num must be >= 0.")

    if args.input_mode == "folder":
        os.makedirs(args.output_dir, exist_ok=True)

    if args.input_mode == "jsonl":
        tasks = build_tasks_from_jsonl(args.jsonl_dir, args.jsonl_files)
    else:
        tasks = build_tasks_from_dirs(args.blended_dir, args.prior_dir)
    if not tasks:
        raise FileNotFoundError("No valid inference tasks found.")
    tasks = filter_pending_tasks(args, tasks, args.noimg)
    if not tasks:
        print("All tasks are already completed.")
        return

    if args.device == "cpu":
        run_worker(args, tasks, "cpu", 0)
        return

    if args.device.startswith("cuda:"):
        run_worker(args, tasks, args.device, 0)
        return

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 0:
        raise RuntimeError("No CUDA devices available.")
    num_gpus = visible_gpu_count if args.num_gpus == 0 else min(args.num_gpus, visible_gpu_count)
    if num_gpus <= 1:
        run_worker(args, tasks, "cuda:0", 0)
        return

    task_chunks = split_tasks_round_robin(tasks, num_gpus)
    processes: List[mp.Process] = []
    for worker_idx, chunk in enumerate(task_chunks):
        if not chunk:
            continue
        process = mp.Process(target=run_worker, args=(args, chunk, f"cuda:{worker_idx}", worker_idx))
        process.start()
        processes.append(process)

    exit_codes = []
    for process in processes:
        process.join()
        exit_codes.append(process.exitcode)
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more worker processes failed: {exit_codes}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
