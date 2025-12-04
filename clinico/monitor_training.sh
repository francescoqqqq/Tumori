#!/bin/bash
# Script per monitorare il training di nnU-Net in tempo reale

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/nnUNet_results/Dataset500_Hybrid_BraTS/nnUNetTrainer__nnUNetPlans__2d/fold_0"
LOG_FILE=$(ls -t "$RESULTS_DIR"/training_log_*.txt | head -1)

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found!"
    exit 1
fi

echo "Monitoring training log: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Watch the log file in real-time
tail -f "$LOG_FILE" | grep -E "(Epoch|train_loss|val_loss|Pseudo dice|learning rate)"

