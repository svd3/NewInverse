import numpy as np
ACTION_DIM = 2
GOAL_RADIUS = 0.5
LOG_REW_WIDTH = np.log(GOAL_RADIUS/2)
TERMINAL_VEL = 0.1
DELTA_T = 0.1
EPSIODE_TIME = 10 # in seconds
EPISODE_LEN = int(EPSIODE_TIME / DELTA_T)
WORLD_SIZE = 2.0
