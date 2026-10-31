#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/self-judge
PY=/root/autodl-tmp/envs/rlvr-overlay/bin/python
RAY=/root/autodl-tmp/envs/rlvr-overlay/bin/ray

BASE="$ROOT/models/Qwen3-VL-8B-Instruct"
DATA="$ROOT/datasets/rlvr_pilot_v1/solver/train_wemath_official_gt_maxpix2m.parquet"
REWARD="$ROOT/rlvr/reward_solver_judge_v1.py"

OUT="$ROOT/outputs/rlvr/solver28_fullbatch28_n16_budget33600_20260925"

###############################################################################
# Experiment definition
###############################################################################

TRAIN_BATCH=28
ROLLOUT_N=16

PPO_MINI=28
PPO_MICRO=4
LOGPROB_MICRO=2

# Higher-utilization rollout config
GPU_MEM_UTIL=0.40
MAX_BATCHED_TOKENS=16384
MAX_NUM_SEQS=256

###############################################################################
# Environment
###############################################################################

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

export RAY_DEDUP_LOGS=0
export RAY_memory_usage_threshold=0.99

# IMPORTANT:
# vLLM 0.20 CuMemAllocator is incompatible with expandable_segments.
unset PYTORCH_CUDA_ALLOC_CONF

unset RAY_ADDRESS
unset RAY_HEAD_IP
unset RAY_REDIS_ADDRESS

###############################################################################
# Banner
###############################################################################

echo "================================================================"
echo "SOLVER28 HIGH-UTILIZATION TEST-1"
echo "================================================================"
echo "DATA                  = $DATA"
echo "TRAIN_BATCH           = $TRAIN_BATCH"
echo "ROLLOUT.N             = $ROLLOUT_N"
echo "TOTAL ROLLOUTS        = $((TRAIN_BATCH * ROLLOUT_N))"
echo
echo "PPO_MINI_BATCH        = $PPO_MINI"
echo "PPO_MICRO_BATCH       = $PPO_MICRO"
echo "LOGPROB_MICRO         = $LOGPROB_MICRO"
echo
echo "GPU_MEMORY_UTIL       = $GPU_MEM_UTIL"
echo "MAX_BATCHED_TOKENS    = $MAX_BATCHED_TOKENS"
echo "MAX_NUM_SEQS          = $MAX_NUM_SEQS"
echo "CHUNKED_PREFILL       = TRUE"
echo "PREFIX_CACHING        = TRUE"
echo "FREE_CACHE_ENGINE     = TRUE"
echo "ENFORCE_EAGER         = TRUE"
echo
echo "TOTAL_STEPS           = 1"
echo "================================================================"

###############################################################################
# Sanity checks
###############################################################################

test -d "$BASE" || {
    echo "FATAL missing BASE: $BASE"
    exit 10
}

test -f "$DATA" || {
    echo "FATAL missing DATA: $DATA"
    exit 11
}

test -f "$REWARD" || {
    echo "FATAL missing REWARD: $REWARD"
    exit 12
}

"$PY" - "$DATA" <<'PY'
import sys
import pandas as pd

p = sys.argv[1]
df = pd.read_parquet(p)

print("rows =", len(df))
print("columns =", list(df.columns))

assert len(df) == 28, f"expected 28 rows, got {len(df)}"

print("SOLVER28_DATA_OK")
PY

###############################################################################
# GPU guard
###############################################################################

GPU_PIDS="$(
    nvidia-smi \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits \
      2>/dev/null \
    | grep -E '^[0-9]+$' || true
)"

if [[ -n "$GPU_PIDS" ]]; then
    echo
    echo "FATAL: GPU IS BUSY"
    echo "$GPU_PIDS"
    nvidia-smi
    exit 30
fi

###############################################################################
# Clean old smoke output / stale Ray
###############################################################################

rm -rf "$OUT"
mkdir -p "$OUT"

"$RAY" stop --force >/dev/null 2>&1 || true

pkill -f 'vLLMHttpServer' 2>/dev/null || true
pkill -f 'EngineCore' 2>/dev/null || true

sleep 5

###############################################################################
# Timing
###############################################################################

START_EPOCH=$(date +%s)

echo
echo "START_TIME=$(date '+%F %T')"
echo

###############################################################################
# One complete GRPO iteration
###############################################################################

set +e

"$PY" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA" \
  data.val_files="$DATA" \
  data.image_key=images \
  data.train_batch_size=28 \
  data.max_prompt_length=6144 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  reward.custom_reward_function.path="$REWARD" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers=1 \
  actor_rollout_ref.model.path="$BASE" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  '+actor_rollout_ref.model.override_config={attn_implementation:sdpa}' \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  'actor_rollout_ref.model.target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=28 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=False \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.disable_log_stats=False \
  trainer.logger='["console"]' \
  trainer.project_name=selfjudge_solver28_fullbatch \
  trainer.experiment_name=solver28_fullbatch28_n16_budget33600 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  'actor_rollout_ref.actor.checkpoint.save_contents=["model"]' \
  +actor_rollout_ref.actor.checkpoint.save_lora_only=True \
  trainer.save_freq=5 \
  trainer.test_freq=-1 \
  trainer.total_epochs=75 \
  trainer.total_training_steps=75 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$OUT"

RC=$?

set -e

END_EPOCH=$(date +%s)
WALL_SECONDS=$((END_EPOCH - START_EPOCH))

###############################################################################
# Cleanup
###############################################################################

"$RAY" stop --force >/dev/null 2>&1 || true

echo
echo "================================================================"
echo "SOLVER28_FULLBATCH_RESULT"
echo "================================================================"
echo "RC=$RC"
echo "WALL_SECONDS=$WALL_SECONDS"
echo "WALL_MINUTES=$((WALL_SECONDS / 60))"
echo "END_TIME=$(date '+%F %T')"
echo "================================================================"

exit "$RC"
