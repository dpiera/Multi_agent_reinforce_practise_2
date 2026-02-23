# Multi-Agent Reinforcement Learning (MARL) Practice

This repository contains implementations and analysis of **Independent Q-Learning (IQL)** and **Centralized Q-Learning (CQL)** applied to different multi-agent environments. The project explores the transition from simple game theory to complex coordination in grid-world scenarios.

---

## 📂 Repository Structure

The project is divided into two logical parts:

### [Part 1: Iterated Prisoner's Dilemma](./Part_1)
Focuses on the fundamental concepts of MARL using the classic Prisoner's Dilemma matrix game.
* **Goal:** Observe the difference between individual greed (Nash Equilibrium) and collective optimization.
* **Key Finding:** IQL agents converge to mutual defection, while CQL successfully learns to cooperate.

### [Part 2: Level-Based Foraging (LBF)](./Part_2)
Extends the algorithms to a high-dimensional grid-world where agents must coordinate to collect food.
* **Goal:** Evaluate performance in competitive vs. cooperative reward structures.
* **Key Finding:** Centralized learning and shared rewards significantly reduce the "moving target" problem, leading to faster convergence and synchronized movement.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the base requirements:
```bash
git clone [https://github.com/dpiera/Multi_agent_reinforce_practise_2.git](https://github.com/dpiera/Multi_agent_reinforce_practise_2.git)
cd Multi_agent_reinforce_practise_2
pip install -r Part_1/requirements.txt
```

### 2. Environment Setup (for Part 2)
To run the Level-Based Foraging experiments, additional dependencies are required:

```bash
pip install gymnasium lbforaging imageio[ffmpeg]
```

### More information on the corresponding folders
