# RL_PID_Tuning
Tuning the PI controller parameters by using a contextual bandit approach.

Requirements.txt includes the packages to be used.

This code tunes the PI controller parameters for a linear system by using the policy gradient while learning a baseline for stability. The environment is assumed to be nonlinear-setpoint-varying. 

States (static/non-Markovian): A steady state operation point. It is also called as setpoint.<br />
Actions (sampled from a continuous policy): PI parameters. D parameter can also be added to the action space.<br />
Reward: Integral squared error + constraints. Note that the constraint weights, W can be tuned to improve safety. <br />

Environment (ThreeTankEnv class): A linear system. This can be substituted by the process of interest. 

User Defined parameters:

W1, W2, W3, W4 (double): Constraint weights for controller gain, time constant, CV, MV.<br />
CONSTRAIN_ALPHA (double): A feasible constraint threshold to keep the tuning process safe.<br />
LR_A, LR_C, CONSTRAIN_LR (double): Learning rates for the policy, critic, constraint.<br /> 
MAX_GLOBAL_EP (integer): Length of episodes.<br />
ENTROPY_BETA (double): Exploration coefficient used in policy gradient.<br />
A_BOUND (list): PI lower and upper bounds.<br />
TRAIN_CTRL (boolean): If true, the agent updates the weights. If false, the agent loads the last policy and tunes the parameters.<br />
self.setpoint: In the 'Worker class', defined by the user.



A reference for full RL (markovian states): https://github.com/stefanbo92/A3C-Continuous/blob/master/a3c.py
