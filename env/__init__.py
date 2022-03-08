import gym

from gym.envs.registration import registry, make, spec
def register(id,*args,**kvargs):
	if id in registry.env_specs:
		return
	else:
		return gym.envs.registration.register(id,*args,**kvargs)

# ------------bullet-------------

register(
	id='hetNetEnv-v0',
	entry_point='env.hetNet:HetNetEnergyEnv',
    max_episode_steps=200,
	reward_threshold=190.0,
)
