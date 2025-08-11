# PPO with Progress-based Entropy Decay – Super Mario Bros

## 📌 Overview
This module trains a **Super Mario Bros** agent using **Proximal Policy Optimization (PPO)** with a **progress-based adaptive entropy coefficient**.

Instead of keeping the entropy coefficient constant, it is **dynamically adjusted** based on the agent's **average maximum horizontal position** over recent episodes using a **logistic decay function**.  
This approach aims to balance exploration and exploitation more effectively as the agent progresses in the environment.

## 🚀 Training
Run the training script:
```bash
python mario_learn.py
```
**Logistic Decay Function**:

```bash
β(p) = β_min + (β_max - β_min) / (1 + exp((p - c) / k))
```

Where:
* p: Average maximum position in recent episodes

* c: Midpoint of decay (controls when the decay is centered)

* k: Smoothness of the decay curve (larger k → slower change)

* β_min, β_max: Minimum and maximum entropy coefficients

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

## 📊 Expected Behavior
* Early training: Higher entropy → encourages exploration.

* Later training: Lower entropy → focuses on exploitation of learned strategies.

* Typically achieves faster convergence and higher peak performance compared to fixed entropy.

