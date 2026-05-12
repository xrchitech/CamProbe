CUDA_VISIBLE_DEVICES=0 \
python -u generate.py \
    --camera_mode custom \
    --motion_path motions/zoom_out.npz \
    --output_path outputs/custom \
    --input_json metadata.jsonl \
    --model_path tencent/HunyuanVideo-1.5

CUDA_VISIBLE_DEVICES=0 \
python -u generate.py \
    --camera_mode latent_bank \
    --output_path outputs/latent_bank \
    --input_json metadata.jsonl \
    --model_path tencent/HunyuanVideo-1.5 \
    --action_id 0 \
    --latent_bank_scale 0.003

CUDA_VISIBLE_DEVICES=0 \
python -u generate.py \
    --camera_mode latent_bank_rotation \
    --output_path outputs/latent_bank_rotation \
    --input_json metadata-rotate.jsonl \
    --model_path tencent/HunyuanVideo-1.5 \
    --action_id 6 \
    --rot_angle 0.02
