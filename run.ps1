# PowerShell equivalent of run.sh
$env:CUDA_VISIBLE_DEVICES = 1
$TAU = $args[0]
$date = Get-Date -Format "yyyy-MMdd"
$time = Get-Date -Format "HHmmss"
$batch_aug = $true
$epochs = 50
$batch_size = 32
$lr = "5e-5"
$seed = 66

# Create directories
New-Item -ItemType Directory -Force -Path "checkpoints/$date"
New-Item -ItemType Directory -Force -Path "log/$date"

# Pretrain
if ($batch_aug) {
    Write-Host "pretrain ps estimator..."
    $save_model = "checkpoints/${date}/${time}_${TAU}_pretrain.pt"
    $log_file = "log/${date}/${time}_${TAU}_pretrain.log"
    $pre_train_args = @(
        "--epochs", $epochs,
        "--batch_size", $batch_size,
        "--learning_rate", $lr,
        "--pre_window", $TAU,
        "--save_model", $save_model,
        "--log_file", $log_file,
        "--seed", $seed,
        "--data_dir", "data/"
    )
    python model/pre_train.py $pre_train_args
}

Write-Host "train main model..."
$outdir = "results_mimic/${date}"
New-Item -ItemType Directory -Force -Path $outdir

$ratio = 0.4
101..150 | ForEach-Object {
    $current_seed = $_
    
    $pretrained_model = "checkpoints/${date}/${time}_${TAU}_pretrain.pt"
    $save_model = "checkpoints/${date}/${time}_${TAU}_${ratio}_${current_seed}.pt"
    $log_file = "log/${date}/${time}_${TAU}_${ratio}_${current_seed}.log"
    $output_mimic = "${outdir}/${time}_${TAU}_${ratio}"
    $output_amsterdamdb = "${outdir}/${time}_${TAU}_${ratio}"

    $main_args = @(
        "--epochs", $epochs,
        "--batch_size", $batch_size,
        "--learning_rate", $lr,
        "--save_model", $save_model,
        "--pretrained_model", $pretrained_model,
        "--log_file", $log_file,
        "--aug_ratio", $ratio,
        "--pre_window", $TAU,
        "--output_mimic", $output_mimic,
        "--output_amsterdamdb", $output_amsterdamdb,
        "--seed", $current_seed,
        "--data_dir", "data/"
    )
    
    python model/main.py $main_args
} 