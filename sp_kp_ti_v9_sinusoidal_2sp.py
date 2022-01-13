""" MSE """
# import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import pickle
import sympy as sym
from scipy import signal
import random

np.random.seed(42)

# PARAMETERS
OUTPUT_GRAPH = False  # save logs
LOG_DIR = './log_pid'  # savelocation for logs
MAX_EP_STEP = 200  # maximum number of steps per episode
MAX_GLOBAL_EP = 20_000  # total number of episodes
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10  # 10  # sets how often the global net is updated
GAMMA = 0.99  # 0.90dont use this  # discount factor
ENTROPY_BETA = 1e-7  # 0.03  # entropy factor
LR_A = 0.0001  # 0.000001  # learning rate for actor
LR_C = 0.001  # 0.00001  # learning rate for critic

N_S = 1  # env.observation_space.shape[0]  # number of states
N_A = 2  # env.action_space.shape[0]  # number of actions
# A_BOUND = [np.array([0.02, 0.02]), np.array([4., 4.])]  # [env.action_space.low, env.action_space.high]  # action bounds
A_BOUND = [np.array([0.02, 0.02]), np.array([2., 2.])]  # [env.action_space.low, env.action_space.high]  # action bounds
# A_BOUND = [np.array([0.1, 0.1]), np.array([10., 10.])]  # [env.action_space.low, env.action_space.high]  # action bounds
#                                                         [np.array([0.01, 0.01]), np.array([1., 1.])]  # action bounds
W1 = 0.025
W2 = 0.025
W3 = 6000.
W4 = 6000.  # weights for kp, taui, CV, MV
CONSTRAIN_ALPHA = 5  # determines determines how much the flow rates can breach in total
CONSTRAIN_LR = 0.000001  # learning rate for the constrain

DISTURBANCE = False
TRAIN_CTRL = True  # Train or not?

if TRAIN_CTRL:
    ISOFFLINE = True
    LOAD_WEIGHTS = False
    SAVE_WEIGHTS = True
    DETERMINISTIC = False
    N_WORKERS = 1  # multiprocessing.cpu_count()  # number of workers
else:
    LOAD_WEIGHTS = True
    DETERMINISTIC = False
    SAVE_WEIGHTS = False
    N_WORKERS = 1
    ISOFFLINE = False

G = 983.991  # cm/s^2
PI = math.pi
prev_best = -180
prev_best_param = [1.2, 15]


# frange is an extension of python's range function that allows for non-integer step sizes
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


# signed square root is the same as a normal square root except the sign of the radicand is preserved
# for example signedSqrt(-4) = -2
def signedSqrt(x):
    if x == 0:
        return 0
    else:
        sign = x / abs(x)
        return sign * abs(x) ** 0.5


class ThreeTankEnv(object):
    def __init__(self, sess, setpoint, isoffline):
        self.sess = sess
        self.setpoint = setpoint  # the list of set points for tank 1

        self.Lambda = 0
        self.C1 = 0  # Kp penalty # bug
        self.C2 = 0  # taui penalty # bug
        self.C3 = 0  # CV penalty # bug
        self.C4 = 0  # MV penalty # bug
        self.breach = 0
        self.constrain_contribution = 0  # constrain to be multiplied by lambda

        self.KP_MAX, self.TAU_MAX, self.MV_MAX, self.CV_MAX = 20, 20, 0.6, self.setpoint * 1.1
        self.KP_MIN, self.TAU_MIN, self.MV_MIN, self.CV_MIN = 0, 0, 0, 0

        self.height_T1_record = []  # list of Tank1 level
        self.flowrate_T1_record = []  # list of Tank1 Flowrate
        self.setpoint_T1_record = []  # list of Tank1 setpoints
        self.kp_record = []  # list of Tank1 Kp
        self.ti_record = []  # list of Tank1 Ti
        self.ep_num = 1  # episode number
        self.old_error1 = 0
        self.new_error1 = 0

        # To calculate MSE
        self.error_sum = 0
        self.no_of_error = 0
        self.time_step = 0  # initial time_step

        # To calculate Variance
        self.flowrate_buffer = []
        self.del_pids = []

        # initialize kp1 and ti1 values
        self.kp1 = 1.2
        self.ti1 = 15
        timespan = np.linspace(0, 100, 101)
        omega = 0.3
        # self.sinfunction = 10 * np.sin(omega * timespan) + 2   # SP varying gain
        # self.sinfunction2 = 15 * np.sin(omega * timespan) + 6  # SP varying tau
        self.sinfunction = 8 * np.sin(omega * timespan) + 2   # SP varying gain
        self.sinfunction2 = 11 * np.sin(omega * timespan) + 6  # SP varying tau
        self.processgain = self.sinfunction[int(setpoint)]
        x = sym.Symbol('x')
        self.processtau = self.sinfunction2[int(setpoint)]
        type2 = sym.Poly((self.processtau * x + 1))
        type2_c = list(type2.coeffs())
        type2_c = np.array(type2_c, dtype=float)
        sys2 = signal.TransferFunction([self.processgain], type2_c)
        sys2 = sys2.to_ss()
        sys2 = sys2.to_discrete(1)
        self.isoffline = isoffline
        if self.isoffline:
            self.A = sys2.A * 0.9
            self.B = sys2.B * 0.9
            self.C = sys2.C * 0.9
        else:
            self.A = sys2.A
            self.B = sys2.B
            self.C = sys2.C

        self.height_T1 = np.asarray([[self.setpoint - 1.]])  # water level of tank 1 in cm
        self.xprime = np.asarray([[self.setpoint - 1.]])
        self.flowrate_T1 = (self.C - self.A) / self.B
        self.state_normalizer = 10.  # 10

    # resets the environment to initial values

    def reinit_the_system(self):
        timespan = np.linspace(0, 100, 101)
        omega = 0.3
        self.sinfunction = 8 * np.sin(omega * timespan) + 2  # 10
        self.sinfunction2 = 11 * np.sin(omega * timespan) + 6  # 15 SP varying tau

        self.processgain = self.sinfunction[int(self.setpoint)]
        x = sym.Symbol('x')
        self.processtau = self.sinfunction2[int(self.setpoint)]

        # self.processtau = 20
        type2 = sym.Poly((self.processtau * x + 1))
        type2_c = list(type2.coeffs())
        type2_c = np.array(type2_c, dtype=float)
        sys2 = signal.TransferFunction([self.processgain], type2_c)
        sys2 = sys2.to_ss()
        sys2 = sys2.to_discrete(1)

        if self.isoffline:
            self.A = sys2.A * 0.9
            self.B = sys2.B * 0.9
            self.C = sys2.C * 0.9
        else:
            self.A = sys2.A
            self.B = sys2.B
            self.C = sys2.C

    def reset_reward(self):
        self.error_sum = 0
        self.no_of_error = 0
        self.flowrate_buffer = []

    def reset(self):
        # This method resets the model and define the initial values of each property
        # self.height_T1 = np.asarray([[0.]])  # Values calculated to be stable at 35% flowrate (below first valve)
        self.height_T1 = np.asarray([[self.setpoint - 1.]]) / self.C  # water level of tank 1 in cm
        self.xprime = np.asarray([[self.setpoint - 1.]]) / self.C
        self.flowrate_T1 = (self.C - self.A) / self.B

        self.Lambda = 0
        self.C1 = 0  # Kp penalty
        self.C2 = 0  # taui penalty
        self.C3 = 0  # CV penalty
        self.C4 = 0  # MV penalty
        self.breach = 0
        self.constrain_contribution = 0  # constrain to be multiplied by lambda

        # initialize PID settings
        self.kp1 = 1.2  # 1.2
        self.ti1 = 15  # 15

        self.time_step = 0  # initial time_step
        self.old_error1 = 0  # initialize errors as zeros
        # normalized error between the water level in tank 1 and the set point
        self.error_sum = 0
        self.no_of_error = 0
        self.flowrate_buffer = []
        error_T1 = self.setpoint - self.height_T1
        self.no_of_error += 1  # Increament the number of error stored by 1
        self.error_sum += np.square(error_T1)  # Sum of error square
        self.new_error1 = error_T1

        self.height_T1_record = []
        self.flowrate_T1_record = []
        self.setpoint_T1_record = []
        self.kp_record = []
        self.ti_record = []

        current_state = [self.setpoint / self.state_normalizer]  # 100. is the max level
        return np.asarray(current_state)

    def update_pid(self, pi_parameters):
        # This method update the pid settings based on the action
        self.kp1 = pi_parameters[0]
        self.ti1 = pi_parameters[1]

    def pid_controller(self):
        # This method calculates the PID results based on the errors and PID parameters.
        # Uses velocity form of the euqation
        del_fr_1 = self.kp1 * (self.new_error1 - self.old_error1 + self.new_error1 / self.ti1)
        del_flow_rate = [del_fr_1]
        # self.flowrate_1_buffer.append(del_fr_1)
        return np.asarray(del_flow_rate)

    def get_setpoints(self):
        return self.setpoint

    # changes the set points
    def set_setpoints(self, setpoints_T1=None):
        if setpoints_T1 is not None:
            self.setpoint = setpoints_T1

    # the environment reacts to the inputted action
    def step(self, delta_flow_rate, disturbance=0):
        # if no value for the valves is given, the valves default to this configuration
        overflow = 0
        pump_bound = 0
        self.flowrate_T1 += delta_flow_rate[0]  # updating the flow rate of pump 1 given the change in flow rate

        if self.flowrate_T1 > 100:
            pump_bound += abs(self.flowrate_T1 - 100)
        elif self.flowrate_T1 < 0:
            pump_bound += abs(self.flowrate_T1)

        if disturbance == 5:
            valves = [1, 1, 1, 1, 1, 1, 1, 0, 1]
        else:
            self.height_T1 = self.height_T1
            valves = [1, 1, 1, 1, 1, 0, 1, 0, 1]

        self.flowrate_T1 = np.clip(self.flowrate_T1, 0, 100)  # bounds the flow rate of pump 1 between 0% and 100%

        setpoint_T1 = self.setpoint

        self.height_T1 = self.xprime
        self.xprime = self.height_T1 * self.A + self.flowrate_T1 * self.B
        self.height_T1 = self.height_T1 * self.C
        self.height_T1 = np.clip(self.height_T1, 0, 43.1)

        if disturbance == 1:
            self.height_T1 = self.height_T1 + 0.1
        elif disturbance == 2:
            self.height_T1 = self.height_T1 + 0.3
        elif disturbance == 3:
            self.height_T1 = self.height_T1 + 0.5
        elif disturbance == 4:
            self.height_T1 = self.height_T1 + 1
        else:
            self.height_T1 = self.height_T1

        if self.kp1 > self.KP_MAX:
            self.C1 = abs(self.kp1 - self.KP_MAX)
        elif self.kp1 < self.KP_MIN:
            self.C1 = abs(self.kp1 - self.KP_MIN)

        if self.ti1 > self.TAU_MAX:
            self.C2 = abs(self.ti1 - self.TAU_MAX)
        elif self.ti1 < self.TAU_MIN:
            self.C2 = abs(self.ti1 - self.TAU_MIN)

        if self.height_T1 > self.CV_MAX:  # MV_MAX
            self.C3 = abs(self.height_T1 - self.CV_MAX)
        elif self.height_T1 < self.CV_MIN:
            self.C3 = abs(self.height_T1 - self.CV_MIN)

        if self.flowrate_T1 > self.MV_MAX:  # MV_MAX
            self.C4 = abs(self.flowrate_T1 - self.MV_MAX)
        elif self.flowrate_T1 < self.MV_MIN:
            self.C4 = abs(self.flowrate_T1 - self.MV_MIN)

        self.constrain_contribution = np.float(abs(W1 * self.C1 + W2 * self.C2 + W3 * self.C3 + W4 * self.C4))

        self.height_T1_record.append(self.height_T1.item())
        self.flowrate_T1_record.append(self.flowrate_T1.item())
        self.setpoint_T1_record.append(setpoint_T1)
        self.kp_record.append(self.kp1)  # store the current kp
        self.ti_record.append(self.ti1)  # store the current ti1

        # calculates the difference between the current water level and its set point in tanks 1 and 3
        # store error as old error since it will be updated soon
        self.old_error1 = self.new_error1
        error_T1 = setpoint_T1 - self.height_T1
        self.no_of_error += 1
        self.error_sum += np.square(error_T1)
        self.new_error1 = error_T1
        # normalizes the heights and errors and returns them as the environment's state
        next_state = [self.setpoint / self.state_normalizer]

        self.time_step += 1  # updates elapsed time
        if self.time_step >= 1000:  # terminates the process if the time elapsed reaches the maximum
            done = True
            self.ep_num += 1
        else:
            done = False
        # returns the next state, reward, and if the episode has terminated or not
        return np.asarray(next_state), done

    def get_reward(self):
        # This method calculates all required factors for reward calculation
        mse = self.error_sum / self.no_of_error  # Sum of error square over the number of errors
        # var_action = np.var(self.flowrate_1_buffer)  # Variance of change in flowrate
        # next_reward_comp = [mse / MSE_MAX, var_action / VAR_MAX, self.breach[0] / EXPLORE_KP,
        #                     self.breach[1] / EXPLORE_TI]  # Normalized based on the max values
        # reward = -W1 * abs(next_reward_comp[0]) - W2 * abs(next_reward_comp[1]) \
        #          - W3 * abs(next_reward_comp[2]) - W4 * abs(next_reward_comp[3])
        reward = - mse.item() * 100 - self.Lambda * self.constrain_contribution
        self.error_sum = 0
        self.no_of_error = 0
        self.flowrate_buffer = []
        return reward


# Network for the Actor Critic
class ACNet(object):
    def __init__(self, scope, sess, globalAC=None):
        self.sess = sess
        self.actor_optimizer = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')  # optimizer for the actor
        self.critic_optimizer = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')  # optimizer for the critic

        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_params, self.c_params = self._build_net(scope)[-2:]  # parameters of actor and critic net
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')  # action
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # v_target value

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(
                    scope)  # get mu and sigma of estimated action from neural net

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    if not DETERMINISTIC:
                        self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0],
                                                  A_BOUND[1])  # sample a action from distribution
                    else:
                        self.A = tf.clip_by_value(mu, A_BOUND[0], A_BOUND[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss,
                                                self.a_params)  # calculate gradients for the network weights
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):  # update local and global network weights
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):  # neural network structure of the actor and critic
        w_init = tf.random_normal_initializer(0., .1)
        b_init = tf.zeros_initializer()

        if LOAD_WEIGHTS:
            # filename = 'Actor_Network_sin_1_baseline.pkl'
            # filename1 = 'Critic_Network_sin_1_baseline.pkl'
            loadnumber = 370701
            filename = f'./Actor_Network_sin_{loadnumber:.0f}.pkl'
            filename1 = f'./Critic_Network_sin_{loadnumber:.0f}.pkl'
            with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
                actor_params_init = pickle.load(f)
                f.close()
            with open(filename1, 'rb') as f1:  # Python 3: open(..., 'rb')
                critic_params_init = pickle.load(f1)
                f1.close()
            w_la = tf.constant_initializer(actor_params_init[0])
            b_la = tf.constant_initializer(actor_params_init[1])
            w_mu = tf.constant_initializer(actor_params_init[2])
            b_mu = tf.constant_initializer(actor_params_init[3])
            w_sigma = tf.constant_initializer(actor_params_init[4])
            b_sigma = tf.constant_initializer(actor_params_init[5])
            w_lc = tf.constant_initializer(critic_params_init[0])
            b_lc = tf.constant_initializer(critic_params_init[1])
            w_v = tf.constant_initializer(critic_params_init[2])
            b_v = tf.constant_initializer(critic_params_init[3])
        else:
            w_la = w_init
            b_la = b_init
            w_mu = w_init
            b_mu = b_init
            w_sigma = w_init
            b_sigma = b_init
            w_lc = w_init
            b_lc = b_init
            w_v = w_init
            b_v = b_init

        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_la, bias_initializer=b_la, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_mu, bias_initializer=b_mu,
                                 name='mu')  # estimated action value
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_sigma, bias_initializer=b_sigma,
                                    name='sigma')  # estimated variance
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_lc, bias_initializer=b_lc, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_v, bias_initializer=b_v,
                                name='v')  # estimated value for state
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.s: s})[0]


# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, name, globalAC, sess):
        global ISOFFLINE
        self.setpoint = 2

        # self.env = ThreeTankEnv(sess, self.setpoint, ISOFFLINE)  # make environment for each worker
        self.name = name
        self.AC = ACNet(name, sess, globalAC)  # create ACNet for each worker
        self.sess = sess

    def work(self):
        global global_rewards, global_constraints, global_episodes, prev_best, prev_best_param
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not coord.should_stop() and global_episodes < MAX_GLOBAL_EP:
            # self.setpoint = random.choice([1, 2])
            # self.setpoint = random.choice([1, 2])
            # self.setpoint = random.choice([4, 5])
            self.setpoint = random.choice([3, 4])

            self.env = ThreeTankEnv(sess, self.setpoint, ISOFFLINE)  # make environment for each worker

            s = self.env.reset()
            self.env.setpoint = self.setpoint - 1

            for i in range(100):
                _, _ = self.env.step(self.env.pid_controller())
            self.env.setpoint = self.setpoint
            self.env.reset_reward(), self.env.reinit_the_system()

            ep_r = 0
            ep_c = 0
            for ep_t in range(MAX_EP_STEP):
                a = self.AC.choose_action(s)  # estimate stochastic action based on policy
                # s_, r, done, info = self.env.step(a)  # make step in environment
                action_multiplier = [5, 5]
                self.env.update_pid(action_multiplier * a)

                for _ in range(1000):
                    s_, _ = self.env.step(self.env.pid_controller())
                done = True
                r = self.env.get_reward()/20
                # done = True if ep_t == MAX_EP_STEP - 1 else False
                print(f'{global_episodes:.0f}| r:{r:.2f}, c:{self.env.constrain_contribution:.2f}',
                      "|", action_multiplier * a)
                ep_r += r
                ep_c += self.env.constrain_contribution
                # save actions, states and rewards in buffer
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize reward

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()  # get global parameters to local ACNet

                s = s_
                total_step += 1
                if done:
                    if len(global_rewards) < 5:  # record running episode reward
                        global_rewards.append(ep_r)
                        global_constraints.append(ep_c)
                    else:
                        global_rewards.append(ep_r)
                        global_rewards[-1] = (np.mean(global_rewards[-5:]))  # smoothing
                        global_constraints.append(ep_c)
                        global_constraints[-1] = (np.mean(global_constraints[-5:]))  # smoothing

                    Loss_c = (ep_c - CONSTRAIN_ALPHA)  # bug
                    # print(f'test{self.env.Lambda + CONSTRAIN_LR * Loss_c}')
                    self.env.Lambda = max(0., self.env.Lambda + CONSTRAIN_LR * Loss_c)  # bug
                    print(f'lambda={self.env.Lambda:.5f}')

                    savenumber = 370711
                    if global_episodes == MAX_GLOBAL_EP - 1:
                        AC_Saved_File = open(f'Reward_sin_{savenumber:.0f}.pkl', 'wb')
                        pickle.dump(global_rewards, AC_Saved_File)
                        AC_Saved_File.close()
                        AC_Saved_File2 = open(f'Constrain_sin_{savenumber:.0f}.pkl', 'wb')
                        pickle.dump(global_constraints, AC_Saved_File2)
                        AC_Saved_File2.close()
                    if global_rewards[-1] > prev_best:
                        prev_best_param = [self.env.kp1, self.env.ti1]
                        prev_best = global_rewards[-1]
                        print(
                            self.name,
                            "Ep:", global_episodes,
                            "| Ep_r: %i" % global_rewards[-1],
                            f'| KP: {self.env.kp1:.2f} taui: {self.env.ti1:.2f}'
                            f"| Best {prev_best:.0f}",
                            f"| Best_params: {prev_best_param[0]:.1f}, {prev_best_param[1]:.1f}"
                        )
                        if SAVE_WEIGHTS:
                            self.AC.pull_global()  # get global parameters to local ACNet
                            saved_actor = self.sess.run(self.AC.a_params)
                            saved_critic = self.sess.run(self.AC.c_params)
                            fileName = f'Actor_Network_sin_{savenumber:.0f}.pkl'
                            fileName1 = f'Critic_Network_sin_{savenumber:.0f}.pkl'
                            AC_Saved_File = open(fileName, 'wb')
                            pickle.dump(saved_actor, AC_Saved_File)
                            AC_Saved_File.close()
                            AC_Saved_File = open(fileName1, 'wb')
                            pickle.dump(saved_critic, AC_Saved_File)
                            AC_Saved_File.close()

                    global_episodes += 1
                    break


if __name__ == "__main__":
    global_rewards = []
    global_constraints = []
    global_episodes = 0

    sess = tf.Session()

    with tf.device("/cpu:0"):
        global_ac = ACNet(GLOBAL_NET_SCOPE, sess)  # we only need its params
        workers = []
        # Create workers
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, global_ac, sess))

    coord = tf.train.Coordinator()
    sess.run(tf.compat.v1.global_variables_initializer())

    if OUTPUT_GRAPH:  # write log file
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.compat.v1.summary.FileWriter(LOG_DIR, sess.graph)

    worker_threads = []
    for worker in workers:  # start workers
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)  # wait for termination of workers

    plt.figure()
    plt.plot(np.arange(len(global_rewards)), global_rewards)  # plot rewards
    plt.xlabel('Episodes')
    plt.ylabel('-MSE')
    # plt.title(f"{GAMMA}")
    plt.title("Episodic Returns")
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(global_constraints)), global_constraints)  # plot rewards
    plt.xlabel('Episodes')
    plt.ylabel('-constraints')
    # plt.title(f"{GAMMA}")
    plt.title("Episodic Constraints")
    plt.show()
