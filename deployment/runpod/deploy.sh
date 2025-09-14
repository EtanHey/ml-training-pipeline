#!/bin/bash

# RunPod Deployment Script

set -e

# Configuration
DOCKER_IMAGE_NAME="ml-inference-runpod"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DOCKER_USERNAME="${DOCKER_USERNAME}"
RUNPOD_API_KEY="${RUNPOD_API_KEY}"
MODEL_PATH="${MODEL_PATH:-./models/model.pth}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}RunPod Deployment Script${NC}"
echo "================================"

# Check required environment variables
if [ -z "$DOCKER_USERNAME" ]; then
    echo -e "${RED}Error: DOCKER_USERNAME not set${NC}"
    exit 1
fi

if [ -z "$RUNPOD_API_KEY" ]; then
    echo -e "${YELLOW}Warning: RUNPOD_API_KEY not set. You'll need to deploy manually via RunPod dashboard${NC}"
fi

# Build Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t ${DOCKER_IMAGE_NAME}:latest .

# Tag for registry
FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${DOCKER_IMAGE_NAME}:latest"
docker tag ${DOCKER_IMAGE_NAME}:latest ${FULL_IMAGE_NAME}

# Push to registry
echo -e "${GREEN}Pushing to Docker registry...${NC}"
docker push ${FULL_IMAGE_NAME}

# Create RunPod deployment configuration
echo -e "${GREEN}Creating RunPod deployment configuration...${NC}"
cat > runpod_config.json <<EOF
{
  "name": "ml-inference-${MODEL_TYPE:-pytorch}",
  "image": "${FULL_IMAGE_NAME}",
  "gpu_type": "NVIDIA GeForce RTX 3090",
  "gpu_count": 1,
  "container_disk_size_gb": 20,
  "volume_size_gb": 10,
  "min_workers": 0,
  "max_workers": 3,
  "env": {
    "MODEL_PATH": "/models/model.pth",
    "MODEL_TYPE": "${MODEL_TYPE:-pytorch}",
    "LOAD_ON_START": "true"
  },
  "scaling": {
    "type": "REQUEST_COUNT",
    "min_workers": 0,
    "max_workers": 3,
    "scale_up_threshold": 1,
    "scale_down_threshold": 0,
    "scale_to_zero_timeout": 300
  }
}
EOF

echo -e "${GREEN}Deployment configuration created: runpod_config.json${NC}"

# Deploy using RunPod CLI if API key is available
if [ ! -z "$RUNPOD_API_KEY" ]; then
    echo -e "${GREEN}Deploying to RunPod...${NC}"

    # Install runpod CLI if not present
    if ! command -v runpodctl &> /dev/null; then
        echo "Installing RunPod CLI..."
        pip install runpodctl
    fi

    # Deploy using CLI
    runpodctl deploy create \
        --name "ml-inference-${MODEL_TYPE:-pytorch}" \
        --image ${FULL_IMAGE_NAME} \
        --gpu-type "NVIDIA GeForce RTX 3090" \
        --gpu-count 1 \
        --min-workers 0 \
        --max-workers 3

    echo -e "${GREEN}Deployment completed!${NC}"
else
    echo -e "${YELLOW}Please deploy manually using the RunPod dashboard:${NC}"
    echo "1. Go to https://runpod.io/console/serverless"
    echo "2. Click 'New Endpoint'"
    echo "3. Use the following Docker image: ${FULL_IMAGE_NAME}"
    echo "4. Configure using the settings in runpod_config.json"
fi

echo -e "${GREEN}Deployment script finished!${NC}"