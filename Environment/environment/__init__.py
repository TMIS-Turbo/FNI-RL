from gym.envs.registration import register

register(id='TrafficEnv1-v0', entry_point='Environment.environment.env1.traffic_env:Traffic_Env')

register(id='TrafficEnv2-v0', entry_point='Environment.environment.env2.traffic_env:Traffic_Env')

register(id='TrafficEnv3-v0', entry_point='Environment.environment.env3.traffic_env:Traffic_Env')

register(id='TrafficEnv4-v0', entry_point='Environment.environment.env4.traffic_env:Traffic_Env')

register(id='TrafficEnv5-v0', entry_point='Environment.environment.env5.traffic_env:Traffic_Env')


