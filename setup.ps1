# setup.ps1
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================="
Write-Host "Setting up Project in Current Directory (MAMBA ONLY)"
Write-Host "============================================="

# Step 1: Check if Mamba is installed
Write-Host "Checking for Mamba..."
if (-not (Get-Command mamba -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Mamba not found. Installing into base..."
    conda install -y -n base -c conda-forge mamba
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install Mamba. Exiting."
        exit 1
    }
} else {
    Write-Host "✅ Mamba already installed."
}

# Step 2: Create folder structure
Write-Host "Creating project directory layout..."
$dirs = @(
    "config",
    "data",
    "data\preprocess",
    "data\loaders",
    "models",
    "train",
    "eval",
    "scripts",
    "checkpoints",
    "logs"
)
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    if (-not (Test-Path $dir)) {
        Write-Host "❌ Failed to create directory '$dir'. Check permissions."
        exit 1
    }
}
Write-Host "✅ Directory structure created."

# Step 3: Create environment.yaml
Write-Host "Creating environment.yaml..."
$envContent = @"
name: moevit
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - transformers
  - einops
  - tqdm
  - matplotlib
  - tensorboard
  - omegaconf
  - opencv
  - black
  - isort
  - jupyter
  - ipykernel
"@
Set-Content -Path "environment.yaml" -Value $envContent
if (-not (Test-Path "environment.yaml")) {
    Write-Host "❌ Failed to create 'environment.yaml'. Check permissions."
    exit 1
}

# Step 4: Create and activate conda environment
Write-Host "Creating Conda environment 'moevit' from environment.yaml..."
mamba env create -f environment.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create environment 'moevit'. Exiting."
    exit 1
}

Write-Host "Activating environment 'moevit'..."
. "$(conda info --base)\Scripts\activate" moevit
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to activate environment 'moevit'. Exiting."
    exit 1
}

# Step 5: Create initial placeholder files
Write-Host "Generating starter files..."

Set-Content -Path "README.md" -Value @"
# Temporal MoE-ViT Project

This repo implements a Mixture-of-Experts Vision Transformer with temporal-aware routing for efficient video understanding.
"@

Set-Content -Path ".gitignore" -Value @"
__pycache__
*.pyc
.DS_Store
*.pt
*.ipynb_checkpoints
logs/
checkpoints/
"@

Set-Content -Path "config\default.yaml" -Value @"
model:
  name: moe_vit
  num_layers: 12
  embed_dim: 768
  num_heads: 12
  num_experts: 8
  top_k: 2
training:
  batch_size: 16
  epochs: 30
  learning_rate: 3e-4
dataset:
  name: TVQA
  video_resolution: 224
  patch_size: 16
  frames: 32
"@

# Create __init__.py and placeholders
$initDirs = @("data", "data\preprocess", "data\loaders", "models", "train", "eval")
foreach ($dir in $initDirs) {
    New-Item -ItemType File -Path "$dir\__init__.py" -Force | Out-Null
    if (-not (Test-Path "$dir\__init__.py")) {
        Write-Host "❌ Failed to create '$dir\__init__.py'. Check permissions."
        exit 1
    }
}

$files = @(
    "data\dataset_factory.py",
    "data\preprocess\extract_patches.py",
    "data\preprocess\tokenize_questions.py",
    "data\loaders\video_text_dataset.py",
    "models\moe_vit.py",
    "models\base_vit.py",
    "models\experts.py",
    "models\router.py",
    "models\attention.py",
    "models\prediction_head.py",
    "train\train.py",
    "train\trainer.py",
    "train\losses.py",
    "eval\evaluate.py",
    "eval\metrics.py",
    "eval\visualize_experts.py",
    "eval\compare_baselines.py",
    "scripts\run_train.bat",
    "scripts\run_eval.bat",
    "scripts\preprocess_tvqa.bat"
)
foreach ($file in $files) {
    New-Item -ItemType File -Path $file -Force | Out-Null
    if (-not (Test-Path $file)) {
        Write-Host "❌ Failed to create '$file'. Check permissions."
        exit 1
    }
}

Write-Host ""
Write-Host "============================================="
Write-Host "✅ Project setup complete in $(Get-Location)!"
Write-Host "➤ Activate env: conda activate moevit"
Write-Host "➤ Environment defined in: environment.yaml"
Write-Host "============================================="