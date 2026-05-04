# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

BASE_MODEL="${BASE_MODEL:-./base_model}"
MODEL_NAME="${MODEL_NAME:-LLaMA-7B}"
ADAPTER_NAME="${ADAPTER_NAME:-DoRA}"
PROMPT_STYLE="${PROMPT_STYLE:-alpaca}"

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_NAME" \
    --dataset boolq \
    --base_model "$BASE_MODEL" \
    --prompt_style "$PROMPT_STYLE" \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/boolq.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_NAME" \
    --dataset piqa \
    --base_model "$BASE_MODEL" \
    --prompt_style "$PROMPT_STYLE" \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/piqa.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_NAME" \
    --dataset social_i_qa \
    --base_model "$BASE_MODEL" \
    --prompt_style "$PROMPT_STYLE" \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/social_i_qa.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_NAME" \
    --dataset hellaswag \
    --base_model "$BASE_MODEL" \
    --prompt_style "$PROMPT_STYLE" \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/hellaswag.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_NAME" \
    --dataset winogrande \
    --base_model "$BASE_MODEL" \
    --prompt_style "$PROMPT_STYLE" \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/winogrande.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_NAME" \
    --dataset ARC-Challenge \
    --base_model "$BASE_MODEL" \
    --prompt_style "$PROMPT_STYLE" \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/ARC-Challenge.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_NAME" \
    --dataset ARC-Easy \
    --base_model "$BASE_MODEL" \
    --prompt_style "$PROMPT_STYLE" \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/ARC-Easy.txt

CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER_NAME" \
    --dataset openbookqa \
    --base_model "$BASE_MODEL" \
    --prompt_style "$PROMPT_STYLE" \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/openbookqa.txt
