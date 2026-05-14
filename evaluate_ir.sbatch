#!/bin/bash
#SBATCH --job-name=tslm_ir_eval
#SBATCH --output=/scratch/users/tgeri/TSLM-CGM/logs/ir_eval_%j.out
#SBATCH --error=/scratch/users/tgeri/TSLM-CGM/logs/ir_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tgeri@stanford.edu

mkdir -p /scratch/users/tgeri/TSLM-CGM/logs

module load python/3.12.1
module load libsndfile
source ~/tslm_env/bin/activate

cd /scratch/users/tgeri/TSLM-CGM

# Load HuggingFace token from .env (never hardcode secrets in scripts)
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | grep HUGGING_FACE_HUB_TOKEN | xargs)
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job $SLURM_JOB_ID started: $(date) ==="
echo "=== Node: $SLURM_NODELIST ==="
nvidia-smi

# MODEL is passed as an env variable, e.g.:
#   sbatch --export=ALL,MODEL=flamingo_pretrained evaluate_ir.sbatch
# Valid values: flamingo_pretrained | flamingo_baseline | sp_pretrained | sp_baseline

case "$MODEL" in
  flamingo_pretrained)
    python -m evaluation.opentslm.evaluate_ir \
      --model_type flamingo \
      --checkpoint hf_models/flamingo_3b_stage5.pt \
      --output results_ir/flamingo_pretrained.jsonl
    ;;
  flamingo_baseline)
    python -m evaluation.opentslm.evaluate_ir \
      --model_type flamingo \
      --output results_ir/flamingo_baseline.jsonl
    ;;
  sp_pretrained)
    python -m evaluation.opentslm.evaluate_ir \
      --model_type sp \
      --checkpoint hf_models/sp_3b_stage5.pt \
      --output results_ir/sp_pretrained.jsonl
    ;;
  sp_baseline)
    python -m evaluation.opentslm.evaluate_ir \
      --model_type sp \
      --output results_ir/sp_baseline.jsonl
    ;;
  *)
    echo "ERROR: Set MODEL env var to one of: flamingo_pretrained, flamingo_baseline, sp_pretrained, sp_baseline"
    exit 1
    ;;
esac

echo "=== Done: $(date) ==="
