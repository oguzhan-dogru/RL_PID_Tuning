# RL_PID_Tuning
The source code for Reinforcement  Learning  Ap-proach to Autonomous PID Tuning.  

Requirements.txt includes the packages to be used.

This contextual-bandit approach (main.py) tunes the PI controller parameters for a linear system by using the policy gradient while learning a baseline for stability. The environment is assumed to be nonlinear-setpoint-varying. 
# Environment Details
States (static/non-Markovian): A steady state operation point. It is also called setpoint.<br />
Actions (sampled from a continuous policy): PI parameters. D parameter can also be added to the action space.<br />
Reward: Integral squared error + constraint violations. Note that the constraint weights, W can be tuned to improve safety. <br />

Environment (ThreeTankEnv class): A linear setpoint-varying system. This can be substituted by the process of interest. 

# User Defined parameters

+ W1, W2, W3, W4 (double): Constraint weights for controller gain, time constant, CV, MV.<br />
+ CONSTRAIN_ALPHA (double): A feasible constraint threshold to keep the tuning process safe.<br />
+ LR_A, LR_C, CONSTRAIN_LR (double): Learning rates for the policy, critic, constraint.<br /> 
+ MAX_GLOBAL_EP (integer): Length of episodes.<br />
+ ENTROPY_BETA (double): Exploration coefficient used in policy gradient.<br />
+ A_BOUND (double list): PI parameter lower and upper bounds.<br />
+ TRAIN_CTRL (boolean): If true, the agent updates the weights. If false, the agent loads the last policy and tunes the parameters.<br />
+ UI (integer): Number of steps in an episode.<br />
+ (KP, TAU, MV, CV)MIN, (KP, TAU, MV, CV)MAX (double): Lower and upper limits for the constraints.<br />
+ self.setpoint (double list in the 'Worker class'): Setpoint list for the PI controller. The agent samples a random setpoint from this list.

Feel free to open issues or send pull requests if you find any bugs. 

# References 
Some references for asynchronous RL, constrained RL, etc.
+ RL-based Tracking with Constrained Filtering: O. Dogru, R. Chiplunkar, and B. Huang, “Reinforcement learning with constrained uncertain reward function through particle filtering,” IEEE Transactions on Industrial Electronics, 2021.
+ Robust Interface Tracking: O. Dogru, K. Velswamy, and B. Huang, “Actor-critic reinforcement learning and application in developing computer-vision-based interface tracking,”
Engineering, 2021. <br />
+ Constrained RL for Low Level Control: O. Dogru, N. Wieczorek, K. Velswamy, F. Ibrahim, and B. Huang, “Online reinforcement learning for a continuous space system with experimental
validation,” Journal of Process Control, vol. 104, pp. 86–100, 2021. <br />

A reference for full RL (markovian states): https://github.com/stefanbo92/A3C-Continuous/blob/master/a3c.py


