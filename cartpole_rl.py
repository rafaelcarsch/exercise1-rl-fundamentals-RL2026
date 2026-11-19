"""
    Robot Learning
    Exercise 1

    Reinforcement Learning 

    Polito A-Y 2026-2027
"""
import torch
import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from agent import Agent, Policy
from utils import get_space_dim

import sys


# Parse script arguments
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--central_point", type=float, default=0.0,
                        help="Point x0 to fluctuate around")
    parser.add_argument("--random_policy", action='store_true', help="Applying a random policy training")
    parser.add_argument("--learning_rate",type=float,default=1e-2)#new: learning rate
    parser.add_argument("--episode_length",type=int,default=500)#new: episode length
    parser.add_argument("--test_episodes", type=int, default=10)#new: number of test episodes   
    parser.add_argument("--file", type=str, default=None) #new: add file argument
    parser.add_argument("--reward", type=str, default=None) #new: add reward argument, specifying the reward approach
    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False,
          silent=False, train_run_id=0, x0=0, random_policy=False, reward_type=None):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state (it's a random initial state with small values)
        observation, _ = env.reset()


        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)

            if random_policy:
                # Task 1.1
                """
                Sample a random action from the action space
                """
                # TODO
                action = env.action_space.sample()

            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            # note that after env._max_episode_steps the episode is over, if we stay alive that long
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # TODO
            if reward_type is not None:
                reward = new_reward(observation, x0, reward_type)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, render=False, x0=0, episode_length=500, reward_type=None):
    test_reward, test_len = 0, 0

    print('Num testing episodes:', episodes)

    for ep in range(episodes):
        done = False
        timesteps = 0
        observation, _ = env.reset()
        osc = {"target": 1.8}
        while not done and timesteps < episode_length:
        # Task 1.2
            """
            Test on 500 timesteps
            """
            # TODO
            #Done by adding episode_length parameter

            action, _ = agent.get_action(observation, evaluation=True)  # Similar to the training loop above -
                                                                        # get the action, act on the environment, save total reward
                                                                        # (evaluation=True makes the agent always return what it thinks to be
                                                                        # the best action - there is no exploration at this point)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # TODO
            if reward_type is not None:
                reward = new_reward(observation, x0, reward_type)

            if render:
                env.render()
            test_reward += reward
            test_len += 1
            timesteps += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)
    return test_reward/episodes


def new_reward(state, x0, type):
    # Task 3.1
    """
        Use a different reward, overwriting the original one
    """
    # TODO
    if type == "CASE1": #Case 1: reward +1 for each timestep, +0.5 for being close to the center
        reward = 1 + 0.5/(1 + abs(state[0]))
    elif type == "CASE2": #Case 2: reward +1 for each timestep, +0.5 for being close to x0
        reward = 1 + 1/(1+abs(state[0]-x0))
    elif type == "CASE3":
        L = 1.8
        x, v = state[0], state[1]
        if abs(x) > L:
            bonus = -np.sign(x) * v               
        else:
            bonus = max(abs(v) - 0.15, 0.0)
        reward = 0.9 + 0.75 * np.clip(bonus, -1.0, 2.0)
    else:
        raise ValueError(f"Unknown reward: {type}")
    return reward   

# The main function
def main(args):
    # Create a Gymnasium environment (CartPole-v1)
    render_mode = "human" if args.render_training or args.render_test else None
    env = gym.make(args.env, render_mode=render_mode, )

    # Task 1.2
    """
    # For CartPole-v1 - change the maximum episode length to 1000
    """
    # TODO
    env._max_episode_steps = args.episode_length

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, args.learning_rate)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        # Train
        training_history = train(agent, env, args.train_episodes, False, args.render_training, x0=args.central_point, random_policy=args.random_policy, reward_type=args.reward)

        # Save the model
        #model_file = "%s_params.ai" % args.env  
        model_file = args.file if args.file is not None else "%s_params.ai" % args.env #new: args.file
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history, color='blue', label='Reward')
        sns.lineplot(x="episode", y="mean_reward", data=training_history, color='orange', label='100-episode average')
        plt.legend()
        plt.title("Reward history (%s)" % args.env)
        plt.show()
        print("Training finished.")
    else:
        # Test
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        global last_test_reward
        last_test_reward = test(agent, env, args.test_episodes, args.render_test, args.central_point, args.episode_length, reward_type=args.reward)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)