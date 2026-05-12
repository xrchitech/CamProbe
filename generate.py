# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import argparse
import copy
import datetime
import json
import os
import shutil

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import einops
import imageio
import loguru
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch import distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.pipelines.my_utils import (
    LatentStepFlowBank,
    LatentStepFlowBank_axis_xyz,
    MiDaSSmall,
)

parallel_dims = initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

LATENT_BANK_ACTION_LABELS = {
    0: "truck_right",
    1: "pan_left",
    2: "tilt_up",
    3: "pedestal_down",
    4: "zoom_in",
    5: "zoom_out",
}

ROTATION_ACTION_LABELS = {
    6: "arc_left",
    7: "arc_right",
}


def load_camera_npz(npz_path, device, ds_scale=1, max_frames=49):
    file_path = npz_path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Camera file not found: {file_path}")

    data = np.load(file_path)
    if "R" not in data or "t" not in data:
        raise ValueError(f"{file_path} does not contain R and t")

    R = data["R"][:max_frames][::ds_scale]
    t = data["t"][:max_frames][::ds_scale]

    R_list = torch.from_numpy(R).float().to(device).unsqueeze(1)
    t_list = torch.from_numpy(t).float().to(device).unsqueeze(1)
    return R_list, t_list


def initialize_latent_bank(bank):
    with torch.no_grad():
        bank.flow.zero_()
        bank.flow[0, 3] = 1.0
        bank.flow[1, 1] = -1.0
        bank.flow[2, 0] = 1.0
        bank.flow[3, 4] = 1.0
        bank.flow[4, 5] = 1.0
        bank.flow[5, 5] = -1.0
    return bank


class I2VPromptDatasetGen(Dataset):
    def __init__(self, json_file):
        self.samples = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        assert len(self.samples) > 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "img_path": sample["image"],
            "prompt": sample["prompt"],
        }


def save_video(video, path):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, "c f h w -> f h w c")
    imageio.mimwrite(path, vid, fps=16)


def rank0_log(message, level):
    if int(os.environ.get("RANK", "0")) == 0:
        loguru.logger.log(level, message)


def save_config(args, output_path, task, transformer_version):
    arguments = {}
    for key, value in vars(args).items():
        if not key.startswith("_") and not callable(value):
            try:
                json.dumps(value)
                arguments[key] = value
            except (TypeError, ValueError):
                arguments[key] = str(value)

    config = {
        "timestamp": datetime.datetime.now().isoformat(),
        "task": task,
        "transformer_version": transformer_version,
        "output_path": output_path,
        "arguments": arguments,
    }

    config_path = os.path.join(output_path, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Saved generation config to: {config_path}")
    return config_path


def str_to_bool(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("true", "1", "yes", "on"):
            return True
        if value in ("false", "0", "no", "off"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def load_checkpoint_to_transformer(pipe, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    rank0_log(f"Loading checkpoint from {checkpoint_path}", "INFO")
    try:
        model_state_dict = get_model_state_dict(pipe.transformer)
        dcp.load(
            state_dict={"model": model_state_dict},
            checkpoint_id=checkpoint_path,
        )
        rank0_log("Transformer model state loaded successfully", "INFO")
    except Exception as exc:
        rank0_log(f"Error loading checkpoint: {exc}", "ERROR")
        raise


def load_lora_adapter(pipe, lora_path):
    rank0_log(f"Loading LoRA adapter from {lora_path}", "INFO")
    try:
        pipe.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_path,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
        rank0_log("LoRA adapter loaded successfully", "INFO")
    except Exception as exc:
        rank0_log(f"Error loading LoRA adapter: {exc}", "ERROR")
        raise


def resolve_rotation_direction(args):
    if args.rotation_direction is not None:
        return args.rotation_direction
    if args.action_id == 6:
        return "left"
    if args.action_id == 7:
        return "right"
    return "left"


def build_warp_configuration(args, device, temporal_ds):
    if args.camera_mode == "custom":
        if not args.motion_path:
            raise ValueError("custom mode requires --motion_path pointing to a single npz file")
        R_list, t_list = load_camera_npz(args.motion_path, device, temporal_ds)
        motion_name = os.path.splitext(os.path.basename(args.motion_path))[0]
        return {
            "warp_mode": "custom",
            "bank": (R_list, t_list),
            "save_tag": motion_name,
        }

    if args.camera_mode == "latent_bank":
        if args.action_id not in LATENT_BANK_ACTION_LABELS:
            raise ValueError(
                f"latent_bank mode requires --action_id in {sorted(LATENT_BANK_ACTION_LABELS)}, got {args.action_id}"
            )
        bank = LatentStepFlowBank(num_dirs=len(LATENT_BANK_ACTION_LABELS)).to(device).eval()
        bank.max_rot = args.latent_bank_scale
        bank.max_trans = args.latent_bank_scale
        initialize_latent_bank(bank)
        action_name = LATENT_BANK_ACTION_LABELS[args.action_id]
        return {
            "warp_mode": "Rt",
            "bank": bank,
            "k": args.action_id,
            "save_tag": f"{args.action_id}_{action_name}",
        }

    if args.camera_mode == "latent_bank_rotation":
        direction = resolve_rotation_direction(args)
        save_name = ROTATION_ACTION_LABELS.get(args.action_id, f"arc_{direction}")
        bank = LatentStepFlowBank_axis_xyz(
            axis=args.rotation_axis,
            direction=direction,
            angle=args.rot_angle,
        ).to(device).eval()
        return {
            "warp_mode": "axis_infer",
            "bank": bank,
            "save_tag": f"{save_name}_{args.rotation_axis}_{args.rot_angle:g}",
        }

    raise ValueError(f"Unsupported camera_mode: {args.camera_mode}")


def generate_video(args):
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), os.path.join(output_dir, os.path.basename(__file__)))

    task = "i2v"
    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        args.resolution,
        task,
        args.cfg_distilled,
        args.enable_step_distill,
        args.sparse_attn,
    )
    if args.save_generation_config:
        save_config(args, output_dir, task, transformer_version)

    infer_state = initialize_infer_state(args)

    if args.sparse_attn and args.use_sageattn:
        raise ValueError("sparse_attn and use_sageattn cannot be enabled simultaneously.")
    if args.use_fp8_gemm and "sgl" in args.quant_type:
        try:
            import sgl_kernel  # noqa: F401
        except Exception as exc:
            raise ValueError(
                "sgl_kernel is not installed. Please install it using `pip install sgl-kernel==0.3.18`"
            ) from exc
    if args.enable_step_distill and args.enable_cache:
        raise ValueError("Enabling both step distilled model and cache will lead to performance degradation.")

    enable_sr = args.sr
    if args.dtype == "bf16":
        transformer_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Must be 'bf16' or 'fp32'")

    enable_offloading = args.offloading
    if args.group_offloading is None:
        offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
        enable_group_offloading = offloading_config["enable_group_offloading"]
    else:
        enable_group_offloading = args.group_offloading
    overlap_group_offloading = args.overlap_group_offloading

    device = torch.device("cpu") if enable_offloading else torch.device("cuda")
    transformer_init_device = torch.device("cpu") if enable_group_offloading else device

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        create_sr_pipeline=enable_sr,
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
    )

    loguru.logger.info(
        f"{enable_offloading=} {enable_group_offloading=} {overlap_group_offloading=}"
    )

    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=enable_offloading,
        enable_group_offloading=enable_group_offloading,
        overlap_group_offloading=overlap_group_offloading,
    )

    if args.checkpoint_path:
        load_checkpoint_to_transformer(pipe, args.checkpoint_path)
    if args.lora_path:
        load_lora_adapter(pipe, args.lora_path)

    if enable_sr and hasattr(pipe, "sr_pipeline"):
        sr_infer_state = copy.deepcopy(infer_state)
        sr_infer_state.enable_cache = False
        pipe.sr_pipeline.apply_infer_optimization(
            infer_state=sr_infer_state,
            enable_offloading=enable_offloading,
            enable_group_offloading=enable_group_offloading,
            overlap_group_offloading=overlap_group_offloading,
        )

    if args.video_length != 121:
        rank0_log(
            f"Warning: 121 frames is the optimal value for best quality. Attempting to generate {args.video_length} frames...",
            "WARNING",
        )
    if not args.rewrite:
        rank0_log(
            "Warning: Prompt rewriting is disabled. This may affect the quality of generated videos.",
            "WARNING",
        )

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    valid_ds = I2VPromptDatasetGen(json_file=args.input_json)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=1,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    midas = MiDaSSmall(
        device,
        depth_mode=args.midas_depth_mode,
        inv_depth_min=args.midas_inv_depth_min,
        depth_min=args.midas_depth_min,
        depth_max=args.midas_depth_max,
        depth_quantile_max=args.midas_depth_quantile_max,
    )
    warp_config = build_warp_configuration(args, device, pipe.vae.ffactor_temporal)

    with torch.no_grad():
        for val_batch in valid_dl:
            prompt = val_batch["prompt"][0]
            img_path = val_batch["img_path"][0]

            extra_kwargs = {
                "reference_image": img_path,
                "start_idx": args.start_idx,
                "end_idx": args.end_idx,
                "warp": True,
                "warp_mode": warp_config["warp_mode"],
                "bank": warp_config["bank"],
                "depth_norm": args.depth_norm,
                "depth_input_source": args.depth_input_source,
                "ablation_mode": args.ablation_mode,
            }
            if "k" in warp_config:
                extra_kwargs["k"] = warp_config["k"]

            print("Generating video ...")
            out = pipe(
                enable_sr=enable_sr,
                prompt=prompt,
                aspect_ratio=args.aspect_ratio,
                num_inference_steps=args.num_inference_steps,
                sr_num_inference_steps=None,
                video_length=args.video_length,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                output_type="pt",
                prompt_rewrite=args.rewrite,
                return_pre_sr_video=args.save_pre_sr_video,
                midas=midas,
                **extra_kwargs,
            )

            input_name = os.path.basename(img_path)
            save_name = os.path.splitext(input_name)[0] + ".mp4"
            save_dir = os.path.join(output_dir, "samples", warp_config["save_tag"])
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, save_name)
            print(f"saving video to {save_path}")
            save_video(out.videos, save_path)


def main():
    parser = argparse.ArgumentParser(description="Generate video using HunyuanVideo-1.5")
    parser.add_argument("--input_json", type=str, default="selected-metadata.jsonl")
    parser.add_argument("--depth_norm", type=str, default="perframe", help="raw, perframe, sequence")
    parser.add_argument(
        "--depth_input_source",
        type=str,
        default="x0pred",
        choices=["latents", "x0pred", "1"],
        help="Depth source during warp: decoded latents, decoded x0 prediction, or constant all-one depth",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default=None,
        choices=["default_v", "eps", "x0", "xt"],
        help="Warp ablation mode inside the diffusion update",
    )
    parser.add_argument(
        "--midas_depth_mode",
        type=str,
        default="relative_inverse_quantile",
        choices=["inverse", "inverse_quantile", "relative", "relative_inverse", "relative_inverse_quantile"],
        help="How to convert MiDaS output into a positive depth proxy",
    )
    parser.add_argument("--midas_inv_depth_min", type=float, default=1e-2)
    parser.add_argument("--midas_depth_min", type=float, default=0.1)
    parser.add_argument("--midas_depth_max", type=float, default=None)
    parser.add_argument("--midas_depth_quantile_max", type=float, default=0.95)
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=5)
    parser.add_argument(
        "--camera_mode",
        "--warp_mode",
        dest="camera_mode",
        type=str,
        default="custom",
        choices=["custom", "latent_bank", "latent_bank_rotation"],
        help="Three supported public modes",
    )
    parser.add_argument(
        "--action_id",
        type=int,
        default=0,
        help="For latent_bank mode: 0=truck_right, 1=pan_left, 2=tilt_up, 3=pedestal_down, 4=zoom_in, 5=zoom_out. For rotation mode: 6=arc_left, 7=arc_right.",
    )
    parser.add_argument(
        "--motion_path",
        type=str,
        default=None,
        help="Single NPZ file path for custom mode",
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        default="motions",
        help="Directory of bundled example NPZ motions",
    )
    parser.add_argument("--latent_bank_scale", type=float, default=0.003, help="Shared rotation/translation magnitude for latent_bank mode")
    parser.add_argument("--rotation_axis", type=str, default="y", choices=["x", "y", "z"])
    parser.add_argument("--rotation_direction", type=str, default=None, choices=["left", "right"])
    parser.add_argument("--rot_angle", type=float, default=0.02)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--video_length", type=int, default=49)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="vivid colors, overexposed, static, blurry details, subtitles, style, artwork, painting, still image, overall gray tone, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, messy background, three legs, crowded background, walking upside down",
    )
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--model_path", type=str, default="tencent/HunyuanVideo-1.5")
    parser.add_argument("--aspect_ratio", type=str, default="16:9")
    parser.add_argument("--sr", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--save_pre_sr_video", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--rewrite", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--cfg_distilled", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--enable_step_distill", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--sparse_attn", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--offloading", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--group_offloading", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--overlap_group_offloading", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--use_sageattn", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--sage_blocks_range", type=str, default="0-53")
    parser.add_argument("--enable_torch_compile", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--enable_cache", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--cache_type", type=str, default="deepcache")
    parser.add_argument("--no_cache_block_id", type=str, default="53")
    parser.add_argument("--cache_start_step", type=int, default=11)
    parser.add_argument("--cache_end_step", type=int, default=45)
    parser.add_argument("--cache_step_interval", type=int, default=4)
    parser.add_argument("--save_generation_config", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--use_fp8_gemm", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--quant_type", type=str, default="fp8-per-token-sgl")
    parser.add_argument("--include_patterns", type=str, default="double_blocks")

    args = parser.parse_args()
    if args.image_path is not None and args.image_path.lower().strip() == "none":
        args.image_path = None

    generate_video(args)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
