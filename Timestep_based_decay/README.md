# PPO with Timestep-based Entropy Decay – Super Mario Bros

## 📌 Overview
This module trains a **Super Mario Bros** agent using **Proximal Policy Optimization (PPO)** with a **timestep-based linear entropy decay**.

Instead of keeping the entropy coefficient constant, it **decreases linearly** as training timesteps increase.  
This method provides a predictable exploration schedule, but may reduce exploration too early in some environments.

## 🚀 Training
Run the training script:
```bash
python mario_learn.py
```

The linear decay function:
```bash
β(t) = max( β_min, β_max + (t / T_max) * (β_min - β_max) )
```
Where:

* t: Current timestep

* T_max: Maximum training timesteps

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
* Early training: Higher entropy → more exploration.

* Later training: Gradual entropy reduction → more exploitation.

* More predictable schedule compared to progress-based decay, but less adaptive to agent’s actual performance.

