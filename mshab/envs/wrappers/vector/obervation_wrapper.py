import gymnasium as gym


class VectorObservationWrapper(gym.vector.VectorEnvWrapper):
    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)

    def observation(self, observation):
        return observation

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self.observation(obs), info

    def step(self, *args, **kwargs):
        obs, rew, term, trunc, info = self.env.step(*args, **kwargs)
        return self.observation(obs), rew, term, trunc, info
