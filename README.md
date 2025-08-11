# RL Exam Project – PPO Entropy Decay Strategies in Super Mario Bros

## 📌 Project Overview
This project investigates how different **entropy coefficient decay strategies** affect the exploration–exploitation balance in **Proximal Policy Optimization (PPO)** when training agents in the *Super Mario Bros* environment.

We compare:
1. **Progress-based decay** – Adapts entropy according to the agent's in-game progress (logistic decay).
2. **Timestep-based decay** – Decreases entropy linearly over total training timesteps.
3. **Fixed coefficient** – Keeps entropy constant throughout training.

Our results show that **progress-based decay** improves learning speed and final performance compared to the other two strategies.

---

## 🎮 Demo
<img src="https://github.com/Simon3999/RL_Exam_Project/blob/main/record.gif" alt="record" style="zoom: 200%;" />

## Install dependencies
```bash
uv sync --python 3.8
pip install -r requirements.txt
```
* Note that the dependencies in requirements.txt need to be reinstalled every time you run ‘uv sync’.
* Also the version of cuda in pyproject.toml should be adjusted to your own situation
