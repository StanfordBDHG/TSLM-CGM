#!/bin/bash
#SBATCH --job-name=tslm_ir_train
#SBATCH --output=/scratch/users/tgeri/TSLM-CGM/logs/ir_train_%j.out
#SBATCH --error=/scratch/users/tgeri/TSLM-CGM/logs/ir_train_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tgeri@stanford.edu

mkdir -p /scratch/users/tgeri/TSLM-CGM/logs

module load python/3.12.1
module load libsndfile
source ~/tslm_env/bin/activate

cd /scratch/users/tgeri/TSLM-CGM

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job $SLURM_JOB_ID started: $(date) ==="
echo "=== Node: $SLURM_NODELIST ==="
nvidia-smi

# MODEL is passed as an env variable:
#   sbatch --export=ALL,MODEL=flamingo_pretrained train_ir.sbatch
# Valid values: flamingo_pretrained | flamingo_baseline | sp_pretrained | sp_baseline

case "$MODEL" in
  flamingo_pretrained)
    python3 curriculum_learning.py \
        --model OpenTSLMFlamingo \
        --stages stage6_ir_cgm_cot \
        --device cuda \
        --llm_id meta-llama/Llama-3.2-3B \
        --results_dir results_ir_pretrained \
        --gradient_checkpointing \
        --batch_size 4 \
        --verbose
    ;;
  flamingo_baseline)
    python3 curriculum_learning.py \
        --model OpenTSLMFlamingo \
        --stages stage6_ir_cgm_cot \
        --device cuda \
        --llm_id meta-llama/Llama-3.2-3B \
        --results_dir results_ir_baseline \
        --gradient_checkpointing \
        --batch_size 4 \
        --verbose
    ;;
  sp_pretrained)
    python3 curriculum_learning.py \
        --model OpenTSLMSP \
        --stages stage6_ir_cgm_cot \
        --device cuda \
        --llm_id meta-llama/Llama-3.2-3B \
        --results_dir results_ir_pretrained \
        --gradient_checkpointing \
        --batch_size 4 \
        --verbose
    ;;
  sp_baseline)
    python3 curriculum_learning.py \
        --model OpenTSLMSP \
        --stages stage6_ir_cgm_cot \
        --device cuda \
        --llm_id meta-llama/Llama-3.2-3B \
        --results_dir results_ir_baseline \
        --gradient_checkpointing \
        --batch_size 4 \
        --verbose
    ;;
  *)
    echo "ERROR: Set MODEL to one of: flamingo_pretrained, flamingo_baseline, sp_pretrained, sp_baseline"
    exit 1
    ;;
esac

echo "=== Done: $(date) ==="
