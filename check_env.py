#!/usr/bin/env python3
import gym
import env

ENV_ID = "hetNetEnv-v0"
RENDER = True


if __name__ == "__main__":
    spec = gym.envs.registry.spec(ENV_ID)
    env = gym.make(ENV_ID)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print(env)
    print(env.reset())
    input("Press any key to exit\n")
    env.closeSim()
