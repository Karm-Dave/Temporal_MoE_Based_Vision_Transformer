# scripts/run_full_project.ps1

Write-Output "============================================================"
Write-Output "========== TEMPORAL-MOE-VIT FULL PROJECT PIPELINE =========="
Write-Output "============================================================"

$env:PYTHONPATH = "."
# IMPORTANT: Make sure you have activated your conda environment
# > conda activate moevit

$CONFIG_FILE = "config/training_msvd.yaml"

# --- STEP 1: PRE-COMPUTATION ---
Write-Output "`n[PHASE 1] Starting Feature Pre-computation (This may take a long time)..."
python scripts/preprocess_features.py
Write-Output "[PHASE 1] Pre-computation complete."


# --- STEP 2: TRAINING ---
# The train.py script now handles data loading and vocab creation robustly.
Write-Output "`n[PHASE 2] Starting Training Phase..."

Write-Output "`n--- TRAINING BASELINE (BaseViT) ON MSVD DATA ---"
python -m train.train --config $CONFIG_FILE --model_type base

Write-Output "`n--- TRAINING OUR MODEL (TemporalMoEViT) ON MSVD DATA ---"
python -m train.train --config $CONFIG_FILE --model_type moe
Write-Output "[PHASE 2] Training complete."


# --- STEP 3: EVALUATION ---
$BASE_CHECKPOINT = "checkpoints/msvd_caption_run_base_best.pt"
$MOE_CHECKPOINT = "checkpoints/msvd_caption_run_moe_best.pt"

Write-Output "`n[PHASE 3] Starting Evaluation Phase..."

Write-Output "`n--- EVALUATING BASELINE (BaseViT) ON VALIDATION SET ---"
python -m eval.evaluate --config $CONFIG_FILE --model_type base --checkpoint_path $BASE_CHECKPOINT

Write-Output "`n--- EVALUATING OUR MODEL (TemporalMoEViT) ON VALIDATION SET ---"
python -m eval.evaluate --config $CONFIG_FILE --model_type moe --checkpoint_path $MOE_CHECKPOINT

Write-Output "`n[PHASE 3] Evaluation complete."
Write-Output "`n--- FULL PROJECT PIPELINE FINISHED ---`n"