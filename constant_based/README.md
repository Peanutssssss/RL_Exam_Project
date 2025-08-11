# PPO with Fixed Entropy Coefficient – Super Mario Bros

## 📌 Overview
This module trains a **Super Mario Bros** agent using **Proximal Policy Optimization (PPO)** with a **constant entropy coefficient** (`ent_coef=0.1`) throughout the entire training process.

It serves as the **baseline** for comparison with other entropy scheduling strategies (progress-based decay, timestep-based decay).

## 🚀 Training
Run the training script:
```bash
python mario_learn.py
```
Best model will be saved in:
```bash
./best_model/best_model.zip
```
Training logs and evaluation results will be stored in:
```bash
./logs/
```

## 🎮 Testing a Trained Model
Run:
```bash
python mario_test.py
```
## 🛠 Utility Functions
utils.py provides:

* make_env() – Creates the wrapped Mario environment with skip frames, grayscale, and resize.

* make_eval_callback() – Evaluates and saves best model during training.

* set_seed() – Ensures reproducible runs.

* quick_seed_test() – Checks if seeding works as expected.

