import gym
from gym.envs.registration import register
register(
    id="airsim-academic-v1", entry_point="airgym.envs:AirSimPursuit",

)
register(
    id="airsim-academic-v2", entry_point="airgym.envs:AirSimAction",

)
register(
    id="airsim-academic-v3", entry_point="airgym.envs:AirSimUAVPursuit",

)
