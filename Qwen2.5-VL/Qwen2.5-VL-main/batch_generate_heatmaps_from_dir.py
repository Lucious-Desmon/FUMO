# batch_generate_heatmaps_from_dir.py
# Process all images in a single folder and save .npy heatmaps to a sibling folder.

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import re
import multiprocessing
from typing import List

# --- ensure qwen_vl_utils.py is available ---
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Error: qwen_vl_utils.py not found or cannot be imported.")
    print("Place it in the same directory or ensure it is on the Python path.")
    raise

# =========================
# Config (edit as needed)
# =========================
INPUT_DIR = "../../blended_images"
OUTPUT_DIRNAME_SUFFIX = "prior"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_PATH = "../Qwen2.5-VL-7B"

# --- heatmap params ---
BOOST_FACTOR = 1.5
BOOST_CAP = 3.8
GUIDED_FILTER_EPS = 0.01**2
KSIZE_PREBLUR = 23

CANDIDATES = ["None", "Minor", "Mid", "Major", "Critical"]
WEIGHTS = {"None": 1, "Minor": 2, "Mid": 3, "Major": 4, "Critical": 5}

# --- visualization params ---
FIXED_MIN_VAL = 1.0
FIXED_MAX_VAL = 4.0
EPS = 1e-6

# resize if long side exceeds this value
MAX_LONG_SIDE = 99999

# =========================
# Path utilities
# =========================
def build_output_path_in_sibling_dir(image_path, input_dir, output_dir, target_ext):
    """
    Output to a sibling folder and keep the same filename with a new extension.
    Example: /a/b/images/123.jpg -> /a/b/imagesprior/123.target_ext
    """
    if not image_path.startswith(os.path.abspath(input_dir)):
        return None
    filename = os.path.splitext(os.path.basename(image_path))[0] + target_ext
    return os.path.join(output_dir, filename)


def list_images_in_dir(input_dir: str) -> List[str]:
    """List images directly under the directory (non-recursive)."""
    files = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(path)
    return files


# =========================
# Core logic (unchanged)
# =========================
def get_candidate_ids(tokenizer):
    ids_map = {}
    for w in CANDIDATES:
        ids = tokenizer(w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            ids_map[w] = ids[0]
        else:
            ids2 = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
            ids_map[w] = ids2[0] if len(ids2) == 1 else ids[0]
    return ids_map


def choose_patch_params(h, w):
    max_side = max(h, w)
    if max_side > 1900:
        return "nonscale", 200, 1.0
    elif 1000 < max_side <= 1900:
        return "nonscale", 170, 1.0
    elif 750 < max_side <= 1000:
        return "nonscale", 130, 1.0
    else:
        return "nonscale", 80, 1.0


@torch.no_grad()
def score_only(pil_img, model, processor, candidate_ids, device):
    prompt = (
        "For this image patch, evaluate the severity of reflection. "
        "Use exactly one of the following words: None, Minor, Mid, Major, Critical. "
        "Reply with only the chosen word, with no extra text."
    )
    messages = [
        {"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True)
    logits = out.scores[0][0]
    sel_logits = torch.stack([logits[candidate_ids[w]] for w in CANDIDATES], dim=0)
    probs = F.softmax(sel_logits, dim=-1)
    weights_tensor = torch.tensor([WEIGHTS[w] for w in CANDIDATES], device=probs.device, dtype=probs.dtype)
    return float(torch.sum(probs * weights_tensor).item())


@torch.no_grad()
def get_reflection_bounding_boxes(img, model, processor, device):
    h, w = img.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    prompt = (
        f"This is an analysis task. The image dimensions are {w}x{h} pixels.\n"
        "In the provided image, please identify and locate any reflections, ghosting, double images, artifacts, "
        "or unnatural light spots and streaks within the image.\n\n"
        "Here are the rules for the output:\n"
        "1. If a single, large, and contiguous area is covered by reflection, please provide one large bounding box.\n"
        "2. If there are multiple, separate, non-contiguous reflection areas, please provide a unique bounding box.\n"
        "3. Ensure boxes are accurate without including non-reflection parts.\n\n"
        "The required output format is a list of lists of four integers, e.g.: [[x1, y1, x2, y2]].\n"
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    input_token_len = inputs["input_ids"].shape[1]
    response_ids = output_ids[0, input_token_len:]
    raw_response = processor.decode(response_ids, skip_special_tokens=True).strip()
    all_coords_matches = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", raw_response)
    return [tuple(map(int, match)) for match in all_coords_matches]


def create_heatmap_for_image(blended_path, prior_path, model, processor, candidate_ids, device):
    img_raw = cv2.imread(blended_path)
    if img_raw is None:
        print(f"[{device}] Warning: failed to read image {blended_path}")
        return

    # record original size
    h0, w0 = img_raw.shape[:2]

    # downscale if needed
    max_side = max(h0, w0)
    if max_side > MAX_LONG_SIDE:
        scale0 = MAX_LONG_SIDE / max_side
        new_w, new_h = int(w0 * scale0), int(h0 * scale0)
        img_for_proc = cv2.resize(img_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_for_proc = img_raw

    h_proc, w_proc = img_for_proc.shape[:2]
    mode, a, scale = choose_patch_params(h_proc, w_proc)
    if mode == "resize":
        img_proc = cv2.resize(img_for_proc, (int(w_proc * scale), int(h_proc * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        img_proc = img_for_proc
    h, w = img_proc.shape[:2]

    # stage 1: patch scores
    pad_h, pad_w = (a - h % a) % a, (a - w % a) % a
    img_pad = cv2.copyMakeBorder(img_proc, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    H, W = img_pad.shape[:2]
    rows, cols = H // a, W // a
    score_grid = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            patch = img_pad[i * a : (i + 1) * a, j * a : (j + 1) * a]
            score = score_only(
                Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)), model, processor, candidate_ids, device
            )
            score_grid[i, j] = score
    score_field_blocky = np.zeros((H, W), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            score_field_blocky[i * a : (i + 1) * a, j * a : (j + 1) * a] = score_grid[i, j]
    score_field_blocky = score_field_blocky[:h, :w]

    # stage 2: detect boxes
    boxes = get_reflection_bounding_boxes(img_proc, model, processor, device)

    # stage 3: boost inside boxes
    enhanced_map = score_field_blocky.copy()
    if boxes:
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            boosted_scores = enhanced_map[y1:y2, x1:x2] * BOOST_FACTOR
            enhanced_map[y1:y2, x1:x2] = np.minimum(boosted_scores, BOOST_CAP)

    # stage 4: guided filter smooth
    radius = int(a * 1.5)
    guide_img_base = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    guide_img = cv2.GaussianBlur(guide_img_base, (KSIZE_PREBLUR, KSIZE_PREBLUR), 0)
    smoothed = cv2.ximgproc.guidedFilter(
        guide=guide_img, src=enhanced_map, radius=radius, eps=GUIDED_FILTER_EPS
    )

    # stage 5: normalize
    clipped = np.clip(smoothed, FIXED_MIN_VAL, FIXED_MAX_VAL)
    denom = max(FIXED_MAX_VAL - FIXED_MIN_VAL, EPS)
    norm = (clipped - FIXED_MIN_VAL) / denom

    # save resized to processed size
    final_h, final_w = img_for_proc.shape[:2]
    norm_resized = cv2.resize(norm, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
    os.makedirs(os.path.dirname(prior_path), exist_ok=True)
    np.save(prior_path, norm_resized.astype(np.float32))

    if "cuda" in str(device):
        torch.cuda.empty_cache()


def worker(tasks, device_id, tokenizer):
    device = f"cuda:{device_id}"
    print(f"Worker started on device {device}")

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16
        ).to(device)
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        candidate_ids = get_candidate_ids(tokenizer)
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"[{device}] Failed to load model: {e}")
        return

    for blended_path, prior_path in tqdm(tasks, desc=f"Worker {device_id}"):
        try:
            create_heatmap_for_image(blended_path, prior_path, model, processor, candidate_ids, device)
        except Exception as e:
            print(f"[{device}] Error processing {blended_path}: {e}")


def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Error: no CUDA devices found.")
        return
    print(f"Found {num_gpus} GPU(s).")

    input_dir = os.path.abspath(INPUT_DIR)
    if not os.path.isdir(input_dir):
        print(f"Error: input dir does not exist: {input_dir}")
        return

    parent_dir = os.path.dirname(input_dir)
    base_name = os.path.basename(input_dir.rstrip(os.sep))
    output_dir = os.path.join(parent_dir, f"{base_name}{OUTPUT_DIRNAME_SUFFIX}")

    image_paths = list_images_in_dir(input_dir)
    if not image_paths:
        print("Error: no images found in input dir.")
        return

    print(f"Found {len(image_paths)} images. Building tasks...")
    all_tasks = []
    for img_path in image_paths:
        prior_path = build_output_path_in_sibling_dir(img_path, input_dir, output_dir, ".npy")
        if not prior_path:
            continue
        if not os.path.exists(prior_path):
            all_tasks.append((img_path, prior_path))

    if not all_tasks:
        print("All tasks already completed.")
        return

    print(f"{len(all_tasks)} new tasks to process.")
    tasks_per_gpu = len(all_tasks) // num_gpus
    task_chunks = [
        all_tasks[i * tasks_per_gpu : (i + 1) * tasks_per_gpu if i < num_gpus - 1 else len(all_tasks)]
        for i in range(num_gpus)
    ]

    print("[Info] Loading tokenizer in main process...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        tokenizer = processor.tokenizer
        print("[Info] Tokenizer loaded.")
    except Exception as e:
        print(f"[Error] Failed to load tokenizer: {e}")
        return

    processes = []
    for i in range(num_gpus):
        if not task_chunks[i]:
            continue
        p = multiprocessing.Process(target=worker, args=(task_chunks[i], i, tokenizer))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\n--- All npy heatmaps generated ---")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
