#!/bin/bash
# Pre-training hook - runs before any training command

echo "🔍 Pre-training checks..."

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.free --format=csv
elif [[ $(uname) == "Darwin" ]] && [[ $(uname -m) == "arm64" ]]; then
    echo "✅ Apple Silicon detected (MPS available)"
else
    echo "⚠️ No GPU detected, will use CPU"
fi

# Check for dataset
if [ -d "datasets" ]; then
    echo "✅ Dataset directory found"
    echo "   Training samples: $(find datasets/train -type f 2>/dev/null | wc -l)"
    echo "   Validation samples: $(find datasets/val -type f 2>/dev/null | wc -l)"
else
    echo "❌ No dataset directory found"
    echo "   Creating dataset structure..."
    mkdir -p datasets/{train,val,test}
fi

# Check for existing experiments
if [ -d "experiments/runs" ]; then
    LAST_RUN=$(ls -t experiments/runs/ 2>/dev/null | head -1)
    if [ ! -z "$LAST_RUN" ]; then
        echo "📊 Last experiment: $LAST_RUN"
        if [ -f "experiments/runs/$LAST_RUN/metrics.jsonl" ]; then
            echo "   Best metric: $(tail -1 experiments/runs/$LAST_RUN/metrics.jsonl | jq -r '.validation.loss')"
        fi
    fi
fi

# Kill any existing training processes
if [ -f ".training.pid" ]; then
    OLD_PID=$(cat .training.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️ Found existing training process (PID: $OLD_PID)"
        read -p "Kill it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill $OLD_PID
            echo "✅ Killed old process"
        fi
    fi
fi

echo "✅ Pre-training checks complete!"