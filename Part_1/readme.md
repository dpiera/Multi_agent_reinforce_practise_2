# Multi-Agent Reinforcement Learning - Part 1: Prisoner's Dilemma

This directory contains the implementation and analysis of two Multi-Agent Reinforcement Learning (MARL) algorithms **Independent Q-Learning (IQL)** and **Centralized Q-Learning (CQL)** applied to the classic Iterated Prisoner's Dilemma.

## 📂 File Structure

* **`iql.py`**: Implementation of the **Independent Q-Learning** agent. Each agent maintains its own Q-table and learns purely from its own rewards, treating the other agent as part of the environment.
* **`cql.py`**: Implementation of the **Centralized Q-Learning** agent. A single central controller learns a joint policy for both agents, optimizing the **sum** of their rewards to achieve cooperation.
* **`train_iql.py`**: Training script for the IQL algorithm. It includes an experiment loop to test different learning rates (0.01, 0.1, 0.5) and visualizes the convergence to the Nash Equilibrium (Mutual Defection).
* **`train_cql.py`**: Training script for the CQL algorithm. It trains the centralized agent and visualizes its convergence to the Global Optimum (Mutual Cooperation).
* **`matrix_game.py`**: Defines the `MatrixGame` environment (gymnasium-based) and the specific payoff matrix for the Prisoner's Dilemma.
* **`utils.py`**: Helper functions for plotting training curves (`visualise_evaluation_returns`), Q-value convergence (`visualise_q_convergence`), and printing Q-tables.
* **`requirements.txt`**: List of Python dependencies required to run the code.

## 🚀 How to Run

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run Independent Q-Learning (IQL)
This script will run experiments with different learning rates and display plots showing that independent agents converge to **Mutual Defection** (Reward -3).

```bash
python train_iql.py
```

### 3. Run Centralized Q-Learning (CQL)
This script will train the centralized agent and display plots showing it learns Mutual Cooperation (Reward -1) by maximizing the joint reward.
```bash
python train_cql.py
```

### 📊 Results Summary
IQL: Agents act greedily and converge to the Nash Equilibrium (Defect, Defect), receiving a reward of -3 each.

CQL: The central controller identifies that (Cooperate, Cooperate) yields the highest total reward (-2 vs -6), successfully solving the dilemma.
