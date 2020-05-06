from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, MlpLstmPolicy, CnnLstmPolicy, CnnPolicy
from differential_gym import DifferentialDriveGym

def visualize():
    n_cpu = 1
    env = SubprocVecEnv([lambda: DifferentialDriveGym() for i in range(n_cpu)])
    model = PPO2.load('model', env=env, tensorboard_log="./pursuit_tensorboard/")

    obs = env.reset()
    while True:
        env.render()
        action, _states = model.predict(obs,deterministic=True)
        print(action[0])
        obs, rewards, dones, info = env.step(action)
        print(dones)
        if dones[0]: obs = env.reset()

def learn():
    n_cpu = 4
    env = SubprocVecEnv([lambda: DifferentialDriveGym() for i in range(n_cpu)]) 
    total_timesteps = 500000
    policy_kwargs = dict(net_arch=[300, 300])
    model = PPO2(MlpPolicy, env, verbose=1, n_steps=128, policy_kwargs=policy_kwargs, tensorboard_log="./no_course_tensorboard/")
    model.learn(total_timesteps=total_timesteps)
    model.save('model')

if __name__ == "__main__":
    visualize()
    # learn()