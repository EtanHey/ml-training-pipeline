#!/bin/bash
# Monitor running training

if [ ! -f ".training.pid" ]; then
    echo "❌ No training process found"
    exit 1
fi

PID=$(cat .training.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ Training process (PID: $PID) is not running"
    rm .training.pid
    exit 1
fi

echo "📊 Training Monitor (PID: $PID)"
echo "================================"

# Get latest log file
LOG_FILE=$(ls -t logs/training_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No log file found"
    exit 1
fi

# Extract metrics from log
echo "📈 Latest Metrics:"
tail -20 "$LOG_FILE" | grep -E "Epoch|Loss|Accuracy|mAP" | tail -5

echo ""
echo "🖥️ System Resources:"

# GPU usage
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU: %s%% | Memory: %s/%s MB\n", $1, $2, $3}'
elif [[ $(uname) == "Darwin" ]]; then
    echo "Apple Silicon GPU (MPS) active"
    # Check process CPU/memory
    ps aux | grep -E "^USER|$PID" | grep -v grep
fi

echo ""
echo "📁 Latest Experiment:"
LATEST_RUN=$(ls -t experiments/runs/ 2>/dev/null | head -1)
if [ ! -z "$LATEST_RUN" ]; then
    echo "Path: experiments/runs/$LATEST_RUN"
    if [ -f "experiments/runs/$LATEST_RUN/metrics.jsonl" ]; then
        echo "Metrics entries: $(wc -l < experiments/runs/$LATEST_RUN/metrics.jsonl)"
    fi
fi

echo ""
echo "🔄 Live Log (last 10 lines):"
echo "----------------------------"
tail -10 "$LOG_FILE"

echo ""
echo "Commands:"
echo "  Stop training:    kill $PID"
echo "  Full log:        tail -f $LOG_FILE"
echo "  Tensorboard:     tensorboard --logdir experiments/runs/"