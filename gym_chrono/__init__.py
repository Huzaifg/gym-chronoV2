import logging
from gym.envs.registration import register


register(
    id='art_wpts-v0',
    entry_point='gym_chrono.envs:art_wpts')  # NAme of the CLASS after the colon
