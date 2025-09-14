# 🚀 Train Your Own AI Model - Super Simple!

No ML experience needed. Just follow these 3 steps.

## Step 1: Get Your Data Ready

Create two folders and add your images:

```
data/
├── train/           # Your training images
│   ├── cats/        # Put cat images here
│   ├── dogs/        # Put dog images here
│   └── birds/       # Put bird images here
└── val/             # A few test images
    ├── cats/        # 5-10 cat images
    ├── dogs/        # 5-10 dog images
    └── birds/       # 5-10 bird images
```

**That's it!** The folder names become your categories.

## Step 2: Train Your Model

Just run:
```bash
./quick_train.sh
```

It will ask you simple questions:
- What type of model? (just press 1 for image classifier)
- Then press 1 to start training

**Training takes about 5-10 minutes on a regular computer.**

## Step 3: Test Your Model

Run the same script again:
```bash
./quick_train.sh
```

Press 3 to test. It opens a webpage where you can upload images!

---

## That's It! 🎉

Your trained model is saved as `model.pt`. You can:
- Share it with friends
- Use it in your apps
- Deploy it online

---

## FAQ

**Q: Do I need a GPU?**
A: No! It works on any computer, just slower without GPU.

**Q: How many images do I need?**
A: At least 20-30 per category. More is better!

**Q: It's not working!**
A: Make sure you have Python installed. Run: `python --version`

**Q: Can I train on something else?**
A: Yes! Just organize your data in folders by category.

---

## Example Projects You Can Build

- 🐕 **Dog Breed Identifier**: Train on different dog breeds
- 🍎 **Fruit Classifier**: Tell apples from oranges
- 😊 **Emotion Detector**: Happy vs sad faces
- 🚗 **Car Model Recognizer**: Different car types
- 🎨 **Art Style Classifier**: Modern vs classical art

---

## One-Line Install (if needed)

```bash
pip install ultralytics torch torchvision gradio
```

That's all you need!

---

## Want More Control?

Check out [README.md](README.md) for advanced features like:
- Training on cloud GPUs
- Fine-tuning hyperparameters
- Deploying to production
- Using different model architectures

But you don't need any of that to get started! 🚀