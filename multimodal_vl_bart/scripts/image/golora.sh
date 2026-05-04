# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

set -euo pipefail

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Missing required environment variable: $name" >&2
        exit 1
    fi
}

require_env VLBART_FEATURE_ROOT
require_env VLBART_DATA_ROOT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
task=multitask
model="bart"

if [ "$model" = "t5" ]; then
    folder_prefix="VLT5"
    backbone="t5-base"
    batch_size=400
else
    folder_prefix="VLBart"
    backbone="facebook/bart-base"
    batch_size=300
fi

feature="${FEATURE_TYPE:-RN101}"
lr="${LR:-1e-3}"
lora_dim="${LORA_DIM:-128}"
theta_clamp="${THETA_CLAMP:-0.10}"

project_name="${feature}_LMsingle_golora_${lora_dim}_bs${batch_size}_image224_lora_settings"
run_name="tune+lr${lr}_theta${theta_clamp}"
output="snap/${folder_prefix}_${task}/${run_name}"

TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node="$1" \
    --master_port="${MASTER_PORT:-26465}" \
    src/${task}.py \
    --distributed --multiGPU \
    --optim adamw \
    --warmup_ratio 0.1 \
    --clip_grad_norm 5 \
    --lr "${lr}" \
    --epochs 20 \
    --num_workers 4 \
    --backbone "${backbone}" \
    --output "${output}" "${@:2}" \
    --num_beams 5 \
    --use_tasks_prompts \
    --batch_size "${batch_size}" \
    --valid_batch_size "${batch_size}" \
    --use_golora \
    --unfreeze_bias \
    --unfreeze_layer_norms \
    --lora_settings \
    --lora_dim "${lora_dim}" \
    --theta_clamp "${theta_clamp}" \
    --orthogonal_direction true \
    --geodesic_direction true \
    --geodesic_detach_trig false \
    --tasks "vqa,gqa,nlvr,caption" \
    --feature "${feature}" --n_boxes 36 --downsample \
    --image_size "(224,224)" \
    --project_name "${project_name}" \
    --run_name "${run_name}"
