#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/self-judge

PYTHON=/root/autodl-tmp/envs/rlvr-overlay/bin/python

MODEL=/root/autodl-tmp/self-judge/models/Qwen3-VL-8B-Instruct

DATA=/root/autodl-tmp/self-judge/datasets/judge_rlvr_243source_v1/cap4/judge_rlvr_cap4.parquet

REWARD=/root/autodl-tmp/self-judge/rlvr/reward_judge_binary_v1.py

EXP=judge_rlvr_243source_cap4_base_n8_20260928

OUT=/root/autodl-tmp/self-judge/outputs/rlvr/${EXP}
ROLLOUT=${OUT}/rollouts

mkdir -p "$OUT"
mkdir -p "$ROLLOUT"
mkdir -p logs

echo "============================================================"
echo "Judge-RLVR Same-Source-243 Cap4"
echo "============================================================"
echo "GPU           : 1 x A800 80GB"
echo "MODEL         : $MODEL"
echo "DATA          : $DATA"
echo "REWARD        : $REWARD"
echo "OUTPUT        : $OUT"
echo
echo "dataset       : 422 Judge prompts"
echo "source Q      : 73 mixed questions from 243-source pool"
echo "families      : 23"
echo "C/F           : 211 / 211"
echo "train batch   : 4"
echo "rollout n     : 8"
echo "temperature   : 0.7"
echo "response len  : 1024"
echo "LoRA          : rank 8 / alpha 16"
echo "lr            : 3e-6"
echo "============================================================"

test -f "$DATA"
test -d "$MODEL"
test -f "$REWARD"

$PYTHON -m py_compile "$REWARD"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

$PYTHON -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="$DATA" \
    data.val_files="$DATA" \
    data.image_key=images \
    data.train_batch_size=4 \
    data.max_prompt_length=6144 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    \
    reward.custom_reward_function.path="$REWARD" \
    reward.custom_reward_function.name=compute_score \
    reward.num_workers=1 \
    \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config="{attn_implementation:sdpa}" \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.actor.checkpoint.save_lora_only=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.26 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    \
    trainer.logger='["console"]' \
    trainer.project_name=selfjudge_judge_rlvr \
    trainer.experiment_name="$EXP" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$OUT" \
    trainer.rollout_data_dir="$ROLLOUT"

