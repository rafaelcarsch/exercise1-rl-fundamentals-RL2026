"""
    Robot Learning
    Exercise 1

    Linear Quadratic Regulator

    Polito A-Y 2026-2027
"""
import gymnasium as gym
import numpy as np
from scipy import linalg     # get riccati solver
import argparse
import matplotlib.pyplot as plt
import sys
from utils import get_space_dim, set_seed
import pdb 
import time

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Environment to use")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--time_sleep", action='store_true',
                        help="Add timer for visualizing rendering with a slower frame rate")
    parser.add_argument("--mode", type=str, default="control",
                        help="Type of test ['control', 'multiple_R']")
    return parser.parse_args(args)

def linerized_cartpole_system(mp, mk, lp, g=9.81):
    mt=mp+mk
    # state matrix
    # a1 = 0
    a1 = (g*mp)/((mk + mp)*(mp/(mk + mp) - 4/3))
    a2 = - g /(lp*((mp/mt)-4/3))
    
    A = np.array([[0, 1, 0,  0],
                  [0, 0, a1, 0],
                  [0, 0, 0,  1],
                  [0, 0, a2, 0]])

    # input matrix
    # b1 = 1/mt
    b1 = -(mp/(mt*(mp/mt - 4/3)) - 1)/mt
    b2 = 1 / (lp*mt*((mp/mt)-4/3))
    B = np.array([[0], [b1], [0], [b2]])
    
    return A, B

def optimal_controller(A, B, R_value=1):
    R = R_value*np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
   # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R),
            np.dot(B.T, P))
    return K

def apply_state_controller(K, x):
    # feedback controller
    # K is 1x4, so Kx is a length-1 array; the controller needs a scalar force
    u = np.asarray(-np.dot(K, x)).item()   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

def multiple_R(env, mp, mk, l, g, time_sleep=False, terminate=None, seed=None):
    """
    Vary R within [0.01, 0.1, 10, 100] and plot applied forces
    over the first 500 timesteps.
    """

    r_list = [0.01, 0.1, 10, 100]
    forces = {}

    A, B = linerized_cartpole_system(mp, mk, l, g)

    for r_value in r_list:

        # Reset environment with the same seed for comparability
        obs, _ = env.reset(seed=seed)

        # Calculate LQR controller
        K = optimal_controller(A, B, R_value=r_value)

        forces[r_value] = []

        for i in range(500):

            if env.render_mode == "human":
                env.render()

            if time_sleep:
                time.sleep(0.1)

            # Calculate action and desired force
            action, force = apply_state_controller(K, obs)

            # Clip force to environment limits
            applied_force = float(np.clip(force, -10, 10))

            # Set force magnitude
            env.unwrapped.force_mag = abs(applied_force)

            # Execute action
            obs, reward, terminated, truncated, _ = env.step(action)

            # Store signed applied force
            forces[r_value].append(applied_force)

            if terminated or truncated:
                print(f"R = {r_value}: Episode ended at timestep {i+1}")
                break

        # Invert forces if the first force is positive
        if forces[r_value] and forces[r_value][0] > 0:
            forces[r_value] = [-f for f in forces[r_value]]

    return forces


def control(env, mp, mk, l, g, time_sleep=False, terminate=True, seed=None):
    """
    Control using LQR
    """
    obs, _ = env.reset(seed=seed)

    A, B = linerized_cartpole_system(mp, mk, l, g)
    K = optimal_controller(A, B)

    env._max_episode_steps = 1000

    states = []

    for i in range(1000):

        if env.render_mode == "human":
            env.render()

        if time_sleep:
            time.sleep(.1)

        # Calculate action and force
        action, force = apply_state_controller(K, obs)

        # Clip force
        abs_force = abs(float(np.clip(force, -10, 10)))

        # Apply force magnitude
        env.unwrapped.force_mag = abs_force

        # Execute action
        obs, reward, terminated, truncated, _ = env.step(action)

        # Store observation
        states.append(obs.copy())

        done = terminated or truncated

        if done:
            print(f'Episode ended after {i+1} total iterations.')

            if terminate:
                break

    return states
# The main function
def main(args):
    # Create a Gymnasium environment (CartPole-v1)
    env = gym.make(args.env, render_mode="None")

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Print some stuff
    print("Environment:", args.env)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    set_seed(args.seed)    # seed for reproducibility

    pole = env.unwrapped
    mp, mk, l, g = pole.masspole, pole.masscart, pole.length, pole.gravity

    if args.mode == "control":
        states_or_forces = control(env, mp, mk, l, g, args.time_sleep, terminate=False, seed=args.seed)
    elif args.mode == "multiple_R":
        states_or_forces = multiple_R(env, mp, mk, l, g, args.time_sleep, terminate=False, seed=args.seed)

    env.close()

    return states_or_forces

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

