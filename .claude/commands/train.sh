#!/bin/bash
# Custom train command for Claude

MODEL_TYPE=${1:-yolo}
MODE=${2:-quick}

# Source pre-training hook
[ -f .claude/hooks/pre-train.sh ] && source .claude/hooks/pre-train.sh

# Set config based on mode
case $MODE in
    quick)
        EPOCHS=10
        CONFIG="experiments/configs/quick.yaml"
        ;;
    full)
        EPOCHS=100
        CONFIG="experiments/configs/full.yaml"
        ;;
    custom)
        EPOCHS=${3:-50}
        CONFIG=${4:-"experiments/configs/custom.yaml"}
        ;;
esac

# Create config if it doesn't exist
if [ ! -f "$CONFIG" ]; then
    mkdir -p $(dirname "$CONFIG")
    cat > "$CONFIG" <<EOF
model_name: ${MODEL_TYPE}v8n
epochs: $EPOCHS
batch_size: 16
learning_rate: 0.001
image_size: 640
device: auto
data_path: ./datasets
monitor_metric: loss
minimize_metric: true
save_interval: 10
early_stopping_patience: 20
EOF
    echo "📝 Created config: $CONFIG"
fi

# Start training in background
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"
mkdir -p logs

echo "🚀 Starting $MODEL_TYPE training ($MODE mode)..."
echo "   Config: $CONFIG"
echo "   Epochs: $EPOCHS"
echo "   Log: $LOG_FILE"

nohup python train_local.py \
    --model-type $MODEL_TYPE \
    --config $CONFIG \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > .training.pid

echo "⏳ Training started (PID: $PID)"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Or check status:"
echo "  .claude/commands/monitor.sh"

# Initial monitoring for 5 seconds
sleep 2
tail -5 "$LOG_FILE"

# Set up trap to run post-hook when training completes
(
    wait $PID
    [ -f .claude/hooks/post-train.sh ] && source .claude/hooks/post-train.sh
) &