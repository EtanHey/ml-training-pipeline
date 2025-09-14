#!/bin/bash
# Quick ML Training - Just run this script!
# Usage: ./quick_train.sh

# Colors for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Quick ML Training Setup ===${NC}"
echo ""

# Check if this is first run
if [ ! -f "config.yaml" ]; then
    echo "First time setup! Let's get you started."
    echo ""

    # Super simple questions
    echo "What are you training?"
    echo "1) Image classifier (recognizes objects/categories)"
    echo "2) Object detector (finds things in images)"
    echo "3) Text model (NLP/chatbot)"
    read -p "Pick one (1-3): " MODEL_TYPE

    case $MODEL_TYPE in
        1)
            MODEL="yolov8n-cls.pt"
            TASK="classify"
            echo "Great! Image classifier selected."
            ;;
        2)
            MODEL="yolov8n.pt"
            TASK="detect"
            echo "Great! Object detector selected."
            ;;
        3)
            MODEL="bert-base"
            TASK="text"
            echo "Great! Text model selected."
            ;;
        *)
            MODEL="yolov8n-cls.pt"
            TASK="classify"
            echo "Defaulting to image classifier."
            ;;
    esac

    # Create minimal config
    cat > config.yaml << EOF
model: $MODEL
task: $TASK
epochs: 10
batch_size: 16
EOF

    # Create data folders
    mkdir -p data/train data/val

    echo ""
    echo -e "${YELLOW}Setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Put your training images in: data/train/"
    echo "2. Put your validation images in: data/val/"
    echo "3. Run this script again to start training"
    echo ""
    read -p "Press Enter when ready..."
fi

# Check if data exists
TRAIN_COUNT=$(find data/train -type f 2>/dev/null | wc -l)
VAL_COUNT=$(find data/val -type f 2>/dev/null | wc -l)

if [ $TRAIN_COUNT -eq 0 ]; then
    echo -e "${YELLOW}No training data found!${NC}"
    echo "Please add images to data/train/ first"
    echo ""
    echo "Example structure:"
    echo "  data/train/cat/cat1.jpg"
    echo "  data/train/cat/cat2.jpg"
    echo "  data/train/dog/dog1.jpg"
    echo "  data/train/dog/dog2.jpg"
    exit 1
fi

echo -e "${BLUE}Found $TRAIN_COUNT training files and $VAL_COUNT validation files${NC}"
echo ""

# Simple menu
echo "What would you like to do?"
echo "1) Train a new model"
echo "2) Continue training existing model"
echo "3) Test your model"
echo "4) Deploy model (share it online)"
read -p "Choice (1-4): " CHOICE

case $CHOICE in
    1)
        echo -e "${GREEN}Starting training...${NC}"

        # Install dependencies if needed
        if ! python -c "import ultralytics" 2>/dev/null; then
            echo "Installing requirements (one-time)..."
            pip install ultralytics torch torchvision gradio --quiet
        fi

        # Simple training command
        echo "Training for 10 epochs (this might take a while)..."
        python -c "
from ultralytics import YOLO
model = YOLO('$(grep model: config.yaml | cut -d' ' -f2)')
model.train(data='data', epochs=10, batch=16)
print('✅ Training complete! Model saved as runs/train/weights/best.pt')
        "

        # Copy to simple location
        cp runs/*/train*/weights/best.pt model.pt 2>/dev/null || cp runs/train/weights/best.pt model.pt
        echo -e "${GREEN}✅ Model saved as model.pt${NC}"
        ;;

    2)
        if [ ! -f "model.pt" ]; then
            echo "No existing model found. Train a new one first!"
            exit 1
        fi

        echo -e "${GREEN}Continuing training from model.pt...${NC}"
        python -c "
from ultralytics import YOLO
model = YOLO('model.pt')
model.train(data='data', epochs=10, batch=16, resume=True)
print('✅ Training complete!')
        "
        ;;

    3)
        if [ ! -f "model.pt" ]; then
            echo "No model found! Train one first."
            exit 1
        fi

        echo -e "${GREEN}Starting test interface...${NC}"
        echo "Opening web browser at http://localhost:7860"

        # Create simple test script
        cat > test_model.py << 'EOF'
import gradio as gr
from ultralytics import YOLO

model = YOLO("model.pt")

def predict(img):
    results = model(img)
    return results[0].plot() if hasattr(results[0], 'plot') else img

gr.Interface(
    fn=predict,
    inputs="image",
    outputs="image",
    title="Test Your Model",
    description="Upload an image to test"
).launch()
EOF

        python test_model.py
        ;;

    4)
        if [ ! -f "model.pt" ]; then
            echo "No model found! Train one first."
            exit 1
        fi

        echo -e "${GREEN}Deployment Options:${NC}"
        echo "1) Hugging Face (free, easy sharing)"
        echo "2) Just give me the model file"
        read -p "Choice (1-2): " DEPLOY_CHOICE

        case $DEPLOY_CHOICE in
            1)
                echo "To deploy to Hugging Face:"
                echo "1. Go to https://huggingface.co/spaces"
                echo "2. Create a new Space"
                echo "3. Upload model.pt and test_model.py"
                echo ""
                echo "Your model file: $(pwd)/model.pt"
                ;;
            2)
                echo -e "${GREEN}Your model is ready at:${NC}"
                echo "$(pwd)/model.pt"
                echo ""
                echo "You can use it in Python:"
                echo "  from ultralytics import YOLO"
                echo "  model = YOLO('model.pt')"
                echo "  results = model('image.jpg')"
                ;;
        esac
        ;;
esac

echo ""
echo -e "${BLUE}Run this script again anytime to continue!${NC}"