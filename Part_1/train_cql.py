import copy
import random
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

# --- CHANGED IMPORT ---
from cql import CQL 
from utils import (
    visualise_q_tables,
    visualise_q_convergence,
    visualise_evaluation_returns,
)
from matrix_game import create_pd_game


CONFIG = {
    "seed": 0,
    "gamma": 0.99,
    "total_eps": 20000,
    "ep_length": 1,
    "eval_freq": 400,
    "lr": 0.1, # Using 0.1 as we found it stable in IQL
    "init_epsilon": 0.9,
    "eval_epsilon": 0.05,
}


def cql_eval(env, config, q_table, eval_episodes=500, output=True):
    """
    Evaluate CQL. Modified to handle the single Q-table structure.
    """
    # --- CHANGED CLASS ---
    eval_agents = CQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["eval_epsilon"],
    )
    # CQL has one q_table, not a list of them
    eval_agents.q_table = q_table

    episodic_returns = []
    for _ in range(eval_episodes):
        obss, _ = env.reset()
        episodic_return = np.zeros(env.n_agents)
        done = False

        while not done:
            actions = eval_agents.act(obss)
            obss, rewards, done, _, _ = env.step(actions)
            episodic_return += rewards

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns, axis=0)
    std_return = np.std(episodic_returns, axis=0)

    if output:
        print("EVALUATION RETURNS:")
        print(f"\tAgent 1: {mean_return[0]:.2f} ± {std_return[0]:.2f}")
        print(f"\tAgent 2: {mean_return[1]:.2f} ± {std_return[1]:.2f}")
    return mean_return, std_return


def train(env, config, output=True):
    # --- CHANGED CLASS ---
    agents = CQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["init_epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["ep_length"]

    evaluation_return_means = []
    evaluation_return_stds = []
    # We store the single q_table here
    evaluation_q_tables = []

    for eps_num in range(config["total_eps"]):
        obss, _ = env.reset()
        episodic_return = np.zeros(env.n_agents)
        done = False

        while not done:
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act(obss)
            n_obss, rewards, done, _, _ = env.step(acts)
            agents.learn(obss, acts, rewards, n_obss, done)

            step_counter += 1
            episodic_return += rewards
            obss = n_obss

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, std_return = cql_eval(
                env, config, agents.q_table, output=output
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            # Store deepcopy of the Q-table
            evaluation_q_tables.append(copy.deepcopy(agents.q_table))

    return (
        evaluation_return_means,
        evaluation_return_stds,
        evaluation_q_tables,
        agents.q_table,
    )
def plot_cql_convergence(q_tables_history):
    """
    Custom plotter to visualize how the Central Agent learns.
    It tracks the value of 'Mutual Cooperation' vs 'Mutual Defection'.
    """
    # Extract Q-values for specific Joint Actions
    # Joint Action 0 = Coop + Coop -> [0, 0]
    # Joint Action 3 = Defect + Defect -> [1, 1]
    
    coop_values = []   # Track value of Cooperation
    defect_values = [] # Track value of Defection
    
    # Obs is always [0, 0] for this matrix game
    obs_key = str([0, 0]) 
    
    for qt in q_tables_history:
        # Key format matches your CQL class: str((obs_str, joint_action_index))
        key_coop = str((obs_key, 0)) # Joint Action 0
        key_defect = str((obs_key, 3)) # Joint Action 3
        
        coop_values.append(qt[key_coop])
        defect_values.append(qt[key_defect])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(coop_values, label="Joint: Cooperate-Cooperate (Action 0)", color='green', linewidth=2)
    plt.plot(defect_values, label="Joint: Defect-Defect (Action 3)", color='red', linestyle='--', linewidth=2)
    
    plt.title("CQL Learning Dynamics: Cooperation vs Defection")
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Q-Value (Expected Sum of Rewards)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    env = create_pd_game()
    
    print("STARTING CQL TRAINING...")
    means, stds, eval_q, final_q = train(env, CONFIG)

    # 1. Plot the Performance (The -1.0 Graph)
    print("Plotting Performance...")
    visualise_evaluation_returns(means, stds)

    # 2. Plot the Q-Convergence (The Brain Scan)
    print("Plotting Q-Values...")
    plot_cql_convergence(eval_q)
 