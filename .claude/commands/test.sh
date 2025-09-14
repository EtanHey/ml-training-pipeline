#!/bin/bash
# Launch interactive test server

MODEL_ARG=${1:-latest}

# Determine model path
if [ "$MODEL_ARG" == "latest" ]; then
    LATEST_RUN=$(ls -t experiments/runs/ 2>/dev/null | head -1)
    MODEL_PATH="experiments/runs/$LATEST_RUN/best_model.pt"
elif [ "$MODEL_ARG" == "best" ]; then
    # Find best model across all runs
    MODEL_PATH=$(find experiments/runs -name "best_model.pt" -exec ls -t {} + | head -1)
else
    MODEL_PATH="$MODEL_ARG"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "Available models:"
    find experiments/runs -name "*.pt" -o -name "*.pth" 2>/dev/null | head -10
    exit 1
fi

echo "🧪 Starting test server..."
echo "   Model: $MODEL_PATH"

# Detect model type from path
if [[ "$MODEL_PATH" == *"yolo"* ]]; then
    MODEL_TYPE="yolo"
elif [[ "$MODEL_PATH" == *"transformer"* ]]; then
    MODEL_TYPE="transformers"
else
    MODEL_TYPE="pytorch"
fi

# Create test samples directory if needed
mkdir -p test_samples
if [ -z "$(ls -A test_samples 2>/dev/null)" ]; then
    echo "📁 test_samples/ is empty"
    echo "   Add test images/data to: $(pwd)/test_samples/"
fi

# Kill any existing test server
if [ -f ".test_server.pid" ]; then
    OLD_PID=$(cat .test_server.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        kill $OLD_PID 2>/dev/null
        sleep 1
    fi
fi

# Start test server
export MODEL_PATH="$MODEL_PATH"
export MODEL_TYPE="$MODEL_TYPE"

echo "🌐 Launching web UI..."
cd deployment/huggingface
python app.py > ../../logs/test_server.log 2>&1 &
PID=$!
cd ../..

echo $PID > .test_server.pid

# Wait for server to start
sleep 3

# Check if server is running
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Test server running!"
    echo ""
    echo "🌐 Web UI: http://localhost:7860"
    echo "📁 Test samples: $(pwd)/test_samples/"
    echo ""
    echo "Quick test commands:"
    echo "  Python:  python -c \"from ultralytics import YOLO; YOLO('$MODEL_PATH')('test_samples/image.jpg').show()\""
    echo "  Stop:    kill $PID"

    # Open browser if possible
    if command -v open &> /dev/null; then
        open http://localhost:7860
    elif command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:7860
    fi
else
    echo "❌ Failed to start test server"
    echo "Check logs: tail -f logs/test_server.log"
    exit 1
fi