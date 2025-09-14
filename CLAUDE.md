# Claude Assistant Guide for ML Training Pipeline

## CLAUDE_COUNTER SYSTEM

**CRITICAL**: Every response MUST include the current CLAUDE_COUNTER value at the end (e.g., "CLAUDE_COUNTER: 7").

- Start at 10
- Decrement by 1 with each response
- When counter reaches 0, immediately re-read the entire CLAUDE.md file before responding
- After re-reading, reset counter to 10

This ensures ongoing alignment with project guidelines and combats drift toward shallow responses.

**IMPORTANT**: After each session compacting/context reset, check the scratchpad file (claude.scratchpad.md) for any relevant context about ongoing training that should be continued.

---

## 🧠 THINKING BEFORE TRAINING (MOST IMPORTANT SECTION)

This section overrides all tendencies toward premature model deployment. When presented with any ML task:

### 1. **Understand the Problem First**

- What's the actual prediction task? Classification, detection, generation?
- What's the business impact? How will this model be used?
- What data do we have? Quality > Quantity always
- What has been tried before? Check `model_history.log` for previous attempts
- **It's okay to say**: "Before I start training, can you help me understand what success looks like?"

### 2. **Explore the Solution Space**

- Think about different model architectures before choosing
- Consider what could go wrong with each approach (overfitting, data leakage, bias)
- Look for existing trained models that could be fine-tuned
- Ask yourself: Do we even need ML for this, or would rules work?
- **Avoid**: Immediately jumping to training without data exploration

### 3. **Practice ML Honesty**

- If the data seems insufficient, say so and explore data collection first
- If you're unsure about hyperparameters, run quick experiments first
- If the model might have ethical implications, discuss them
- Share your thought process - the reasoning matters more than the result
- **Remember**: It's better to collect more data than to overfit on limited data

### 4. **Stay Iterative and Experimental**

- Why does this particular metric matter to the user?
- What can you learn from the failure modes?
- What edge cases should we test for?
- How might this model drift in production?
- **It's natural to**: Need multiple training iterations to get good results

### Common Anti-Patterns to Avoid:

- ❌ Training without checking data distribution first
- ❌ Deploying the first model that "works"
- ❌ Using complex models when simple ones would suffice
- ❌ Assuming more epochs = better model
- ❌ Ignoring validation metrics in favor of training metrics

---

## SCRATCHPAD FOR COMPLEX TRAINING

A file called `claude.scratchpad.md` exists at the repository root for tracking training experiments:
- Track hyperparameter experiments and their results
- Store intermediate metrics for comparison
- Plan multi-stage training pipelines
- Document data quality issues discovered
- The scratchpad should persist across messages but be cleared after deployment

Example scratchpad entry:
```markdown
## Experiment Log - YOLO Hand Detection

### Attempt 1 (10 epochs, batch=16)
- Train loss: 0.23
- Val loss: 0.45 (overfitting!)
- mAP: 0.67

### Attempt 2 (10 epochs, batch=16, dropout=0.5)
- Train loss: 0.31
- Val loss: 0.33 (better!)
- mAP: 0.71

Next: Try data augmentation...
```

---

## 🎯 Real-World ML Training Workflow

**Train Locally → Test Interactively → Deploy When Ready**

Unlike traditional CI/CD, ML requires human judgment at key checkpoints. This pipeline embraces that reality.

## 📊 Training Best Practices

### Data Quality Checks (ALWAYS DO FIRST)

```python
# Before ANY training, check:
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Class distribution: {class_counts}")

# Red flags:
- Imbalanced classes (>10:1 ratio)
- Too few samples (<100 per class)
- Validation set too small (<20% of training)
- Duplicate images
- Data leakage (test samples in training)
```

### Model Versioning Strategy

Following proven patterns from production:

```bash
# Automatic version numbering (like finetune-workshop)
model-v1.pt  # Initial training
model-v2.pt  # After adding more data
model-v3.pt  # After hyperparameter tuning

# Each version tracks in model_history.log:
echo "$(date): v3 - Added augmentation, fixed overfitting, 94% accuracy" >> model_history.log
```

### Background Training Management

When user says "train a model":

1. **Check resources first**:
```bash
# GPU check
nvidia-smi || echo "Using CPU/MPS"
# Kill old training
[ -f .training.pid ] && kill $(cat .training.pid)
```

2. **Start training in background**:
```bash
python train.py --config config.yaml > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > .training.pid
```

3. **Monitor automatically**:
```bash
# Show live metrics
tail -f logs/training_*.log | grep -E "Epoch|Loss|mAP"
```

---

## 🔧 Common Adjustments During Training

### When Model Overfits
```python
# Claude detects: val_loss increasing while train_loss decreasing
# Auto-suggests:
config['dropout'] = 0.5  # Add dropout
config['weight_decay'] = 1e-4  # Add L2
config['augmentation'] = True  # Enable augmentation
# ALSO: Check if you need more diverse training data
```

### When Training Stalls
```python
# Claude detects: loss plateau for 5+ epochs
# Auto-suggests:
config['learning_rate'] *= 0.1  # Reduce LR
# OR: Switch to cosine annealing
# OR: Check if model capacity is too small
```

### When GPU OOM
```python
# Claude detects: CUDA out of memory
# Auto-adjusts:
config['batch_size'] //= 2  # Halve batch size
config['gradient_accumulation'] = 2  # Maintain effective batch
# OR: Use smaller model variant
```

---

## RunPod Integration (Direct SSH Pattern)

Based on real-world usage that actually works:

### Setup (One-time)
```bash
# Create SSH key without passphrase
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_runpod -N ""

# Add to RunPod before creating pod
cat ~/.ssh/id_ed25519_runpod.pub  # Copy this to RunPod settings
```

### Connection That Works
```bash
# Use direct IP from RunPod dashboard (NOT proxy)
ssh -p [PORT] root@[IP] -i ~/.ssh/id_ed25519_runpod

# Upload data (works reliably)
scp -r -P [PORT] -i ~/.ssh/id_ed25519_runpod datasets/ root@[IP]:/workspace/

# NOT this (often fails):
# ssh [pod-id]@ssh.runpod.io
```

### Training on RunPod
```bash
# IMPORTANT: Keep YOLO commands on ONE LINE
yolo classify train model=yolov8n.pt data=/workspace/datasets epochs=15 batch=32 device=0

# Download results
scp -P [PORT] -i ~/.ssh/id_ed25519_runpod root@[IP]:/workspace/runs/classify/train/weights/best.pt ./model_v2.pt
```

---

## 📦 Deployment Decision Tree

```
Is model ready?
├── Quick Demo → Hugging Face Spaces
│   └── No Docker needed, instant web UI
├── Production API → RunPod
│   └── Docker + GPU autoscaling
├── Edge/Browser → TensorFlow.js
│   └── Privacy-preserving client-side
└── Continue Training → Adjust and retry
```

---

## Project Structure (Minimal, No Deadweight)

```
your-ml-project/
├── train.py           # Single training script
├── config.yaml        # Configuration
├── datasets/          # Your data
│   ├── train/
│   └── val/
├── models/            # Saved models (v1, v2, etc.)
├── logs/              # Training logs
├── model_history.log  # What worked, what didn't
└── claude.scratchpad.md  # Experiment tracking
```

---

## ML-Specific Note Guidelines

Use ML-specific anchors for complex training code:
- `MLDEV-NOTE:` - Important training insights
- `MLDEV-TODO:` - Planned experiments
- `MLDEV-RESULT:` - Experiment outcomes
- `MLDEV-BUG:` - Known issues with training

Example:
```python
# MLDEV-NOTE: Batch size >32 causes OOM on RTX 3090
# MLDEV-RESULT: Dropout=0.5 reduced overfitting by 40%
# MLDEV-TODO: Try mixed precision training for speedup
```

---

## 🚨 When Things Go Wrong

### Training Crashes
```bash
# Check last error
tail -100 logs/training_*.log | grep -i error

# Common fixes:
- Reduce batch size (OOM)
- Check data paths exist
- Verify CUDA compatibility
- Check for corrupt images in dataset
```

### Model Performs Poorly
```python
# Quick diagnostics
# 1. Check class distribution
# 2. Visualize predictions on validation set
# 3. Look for systematic failures
# 4. Check if data augmentation is too aggressive
```

### Can't Connect to RunPod
```bash
# Always use direct IP, not proxy
ssh -p [PORT] root@[IP] -i ~/.ssh/runpod_key

# If "subsystem request failed", you're using proxy - switch to IP
```

---

## 🎮 Interactive Commands

When working with this pipeline, Claude responds to:

- **"train"** → Start background training with monitoring
- **"status"** → Show current training progress
- **"test"** → Launch interactive test server
- **"compare"** → Compare last two model versions
- **"deploy"** → Guide through deployment options
- **"diagnose"** → Analyze why model is failing
- **"data"** → Analyze dataset quality
- **"stop"** → Safely stop training

---

## 📚 Key Principles

1. **Data Quality > Model Complexity** - Better data beats fancy architectures
2. **Test Incrementally** - Train for few epochs first, validate approach
3. **Version Everything** - Models, data, configs, results
4. **Document What Works** - Future you will thank current you
5. **Human Judgment Matters** - Not everything can be automated

---

## Modular Setup Script

Use `setup.sh` to create minimal projects with only needed components:
```bash
./setup.sh
# Select: 1) YOLO  2) None for deployment  3) TensorBoard  4) Local only
# Creates minimal ~50KB project instead of bloated pipeline
```

Or use the single-file approach:
```bash
# Just copy ml.py to any project
cp ml.py ~/my-new-project/
cd ~/my-new-project
python ml.py train
```

---

*This guide is based on real-world ML training patterns that actually work in production, not theoretical best practices that look good on paper but fail in practice.*

CLAUDE_COUNTER: 10