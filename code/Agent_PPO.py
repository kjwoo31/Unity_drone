# submission을 위한 Agent 파일 입니다.
# policy(), save_model(), load_model()의 arguments와 return은 변경하실 수 없습니다.
# 그 외에 자유롭게 코드를 작성 해주시기 바랍니다.

import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from mlagents_envs.environment import ActionTuple
from Drone_gym import Drone_gym

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()


        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            # nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)

        over = torch.max(action_probs)
        action_probs = action_probs - over.expand_as(action_probs)
        action_probs = nn.functional.log_softmax(action_probs, dim=-1)

        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)

        over = torch.max(action_probs)
        action_probs = action_probs - over.expand_as(action_probs)
        action_probs = nn.functional.log_softmax(action_probs, dim=-1)

        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class Agent:
    def __init__(self, state_dim=26, action_dim=7, lr_actor=0.0003, lr_critic=0.001, gamma=0.95, K_epochs=30, eps_clip=0.2, action_std_init=0.6):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy_ac = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy_ac.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy_ac.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy_ac.state_dict())

        self.MseLoss = nn.MSELoss()

    def policy(self, state):
        """Policy function p(a|s), Select three actions.

        Args:
            obs: Only robot, location, lidar information from the vector observations.
                   (vector observation)

        Return:
            27 discrete actions vector (range 0 ~ 26) from policy.
            ex) [-1, -1, -1], [0, -1, -1], ...
        """

        with torch.no_grad():
            #order = self.best_order(state[5][9:12], state[5][:3], state[5][3:6], state[5][6:9])
            #state, curriculum = self.convert_obs(state, order)

            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)

        action = self.convert_action(action.item())

        return action

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        if rewards == [0.0]:
            print(self.buffer.rewards)
            print("strange reward")
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards.resize_(rewards.size(dim=0))
        rewards = torch.nan_to_num(rewards)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy_ac.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy_ac.state_dict())

        # clear buffer
        self.buffer.clear()

    def save_model(self):
        checkpoint_path = "best_model/best_model.pt"
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        return None

    def load_model(self, weight_file_path="best_model/best_model.pt", map_location=None):
        checkpoint_path = weight_file_path
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy_ac.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        return None

    def convert_obs(self, obs, order):
        """
           set initial observation
        """
        final_obs = []
        # target_obs = obs[5][:9]  # 9
        # print(obs[5][18])
        if obs[5][18] == 1.0:
            target_obs = order[0] #3
            curriculum = 0
        elif obs[5][18] < 0.4:
            target_obs = order[2] #3
            curriculum = 2
        else:
            target_obs = order[1] #3
            curriculum = 1
        robot_obs = obs[5][9:18] #9
        lidar_obs = obs[5][19:43] #24
        lidar_obs = np.append(lidar_obs, obs[5][91:95]) #4
        # only_lidar_obs = lidar_obs[1::2] #38

        final_obs = np.append(final_obs, target_obs)
        final_obs = np.append(final_obs, robot_obs)
        for i in range(14): # len(lidar_obs)/2 = 12 + 2
            if lidar_obs[2*i] == 1:
                final_obs = np.append(final_obs, 0)
            else:
                final_obs = np.append(final_obs, lidar_obs[2*i + 1])
        # final_obs = np.append(final_obs, only_lidar_obs)

        return final_obs, curriculum

    def convert_action(self, action):
        if action == 0:
            return [0.5, 0, 0]
        elif action == 1:
            return [0, 0.5,  0]
        elif action == 2:
            return [0, 0,  0.5]
        elif action == 3:
            return [-0.5,  0, 0]
        elif action == 4:
            return [0,  -0.5,  0]
        elif action == 5:
            return [0,  0,  -0.5]
        else:
            return [0, 0, 0]

    # def convert_action(self, action): # 26
    #     if action == 0:
    #         return [-1, -1, -1]
    #     elif action == 1:
    #         return [-1, -1,  0]
    #     elif action == 2:
    #         return [-1, -1,  1]
    #     elif action == 3:
    #         return [-1,  0, -1]
    #     elif action == 4:
    #         return [-1,  0,  0]
    #     elif action == 5:
    #         return [-1,  0,  1]
    #     elif action == 6:
    #         return [-1,  1, -1]
    #     elif action == 7:
    #         return [-1,  1,  0]
    #     elif action == 8:
    #         return [-1,  1,  1]
    #     elif action == 9:
    #         return [ 0, -1, -1]
    #     elif action == 10:
    #         return [ 0, -1,  0]
    #     elif action == 11:
    #         return [ 0, -1,  1]
    #     elif action == 12:
    #         return [ 0,  0, -1]
    #     elif action == 13:
    #         return [ 0,  0,  1]
    #     elif action == 14:
    #         return [ 0,  1, -1]
    #     elif action == 15:
    #         return [ 0,  1,  0]
    #     elif action == 16:
    #         return [ 0,  1,  1]
    #     elif action == 17:
    #         return [ 1, -1, -1]
    #     elif action == 18:
    #         return [ 1, -1,  0]
    #     elif action == 19:
    #         return [ 1, -1,  1]
    #     elif action == 20:
    #         return [ 1,  0, -1]
    #     elif action == 21:
    #         return [ 1,  0,  0]
    #     elif action == 22:
    #         return [ 1,  0,  1]
    #     elif action == 23:
    #         return [ 1,  1, -1]
    #     elif action == 24:
    #         return [ 1,  1,  0]
    #     elif action == 25:
    #         return [ 1,  1,  1]

    def best_order(self, o, a1, a2, a3):
        dist_list = {
            np.linalg.norm(o - a1) + np.linalg.norm(a1 - a2) + np.linalg.norm(a2 - a3),  # 0,1,2,3
            np.linalg.norm(o - a1) + np.linalg.norm(a1 - a3) + np.linalg.norm(a3 - a2),  # 0,1,3,2
            np.linalg.norm(o - a2) + np.linalg.norm(a2 - a1) + np.linalg.norm(a1 - a3),  # 0,2,1,3
            np.linalg.norm(o - a2) + np.linalg.norm(a2 - a3) + np.linalg.norm(a3 - a1),  # 0,2,3,1
            np.linalg.norm(o - a3) + np.linalg.norm(a3 - a1) + np.linalg.norm(a1 - a2),  # 0,3,1,2
            np.linalg.norm(o - a3) + np.linalg.norm(a3 - a2) + np.linalg.norm(a2 - a1),  # 0,3,2,1
        }

        if np.argmin(dist_list) == 1:
            order = np.array((a1, a3, a2))
        elif np.argmin(dist_list) == 2:
            order = np.array((a2, a1, a3))
        elif np.argmin(dist_list) == 3:
            order = np.array((a2, a3, a1))
        elif np.argmin(dist_list) == 4:
            order = np.array((a3, a1, a2))
        elif np.argmin(dist_list) == 5:
            order = np.array((a3, a2, a1))
        else:
            order = np.array((a1, a2, a3))

        return order

def main():
    env = Drone_gym(
        time_scale=1.0,
        port=11000, #width=84, height=84,
        filename='../RL_Drone/drone.x86_64')
    env_name = "w"

    '''
    해상도 변경을 원할 경우, width, height 값 조절.
    env = Drone_gym(
            time_scale=1.0,
            port=11000,
            width=84, height=84, filename='../RL_Drone/DroneDelivery.exe')
    '''

    episode = 500000

    print_freq = 100  # print avg reward in the interval (in num timesteps)
    log_freq = 1  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1)  # save model frequency (in num timesteps)

    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################

    K_epochs = 30  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    #####################################################

    # state space dimension
    state_dim = 26

    # action space dimension
    action_dim = 7

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    #####################################################

    ################### checkpointing ###################

    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    #####################################################

    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = Agent()
    # try:
    #     epitime=50000
    #     checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, epitime, run_num_pretrained)
    ppo_agent.load_model() # user의 모델 불러오기. 경로는 best_model 폴더.
    #     print("Succeed load User's model")
    # except:
    #     print("Fail load User's model'")
    #     raise


    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0

    # training loop
    for epi in range(episode):
        obs = env.reset()
        order = ppo_agent.best_order(obs[5][9:12], obs[5][:3], obs[5][3:6], obs[5][6:9])
        state, curriculum = ppo_agent.convert_obs(obs, order)
        current_ep_reward = 0

        while True:
            # # normlaize
            # norm = np.linalg.norm(state)
            # if norm != 0:
            #     state = state / norm

            # select action with policy
            action = ppo_agent.policy(state)

            action_tuple = ActionTuple()
            action_tuple.add_continuous(np.array([action]))
            env.env.set_actions(env.behavior_name, action_tuple)

            # 행동 수행
            env.env.step()

            # 행동 수행 후 에이전트의 정보 (상태, 보상, 종료 여부) 취득
            decision_steps, terminal_steps = env.env.get_steps(env.behavior_name)

            done = len(terminal_steps.agent_id) > 0

            if done:
                next_state = [terminal_steps.obs[i][0] for i in range(6)]
            else:
                next_state = [decision_steps.obs[i][0] for i in range(6)]

            reward = 0
            time_step += 1

            find_obstacle = np.where(state == 0, 2, state)
            if min(find_obstacle[12:]) < 0.5:
                arg = np.argmin(find_obstacle[12:])

                next_lidar_state = []
                lidar_obs = next_state[5][19:43]  # 24
                lidar_obs = np.append(lidar_obs, next_state[5][91:95])  # 4
                for i in range(14):  # len(lidar_obs)/2 = 12 + 2
                    if lidar_obs[2 * i] == 1:
                        next_lidar_state = np.append(next_lidar_state, 0)
                    else:
                        next_lidar_state = np.append(next_lidar_state, lidar_obs[2 * i + 1])

                else:
                    if arg > 11:
                        if next_lidar_state[arg] == 0:
                            reward += 2 - state[12:][arg]
                        else:
                            reward += next_lidar_state[arg] - state[12:][arg]
                    else:
                        if action == 2:
                            reward = 1 + next_lidar_state[arg] - state[12:][arg]
                        else:
                            reward = -1 + next_lidar_state[arg] - state[12:][arg]
            else:
                if curriculum == 0:
                    reward -= (np.linalg.norm(order[0] - next_state[5][9:12]) - np.linalg.norm(order[0] - state[3:6]))
                elif curriculum == 2:
                    reward -= (np.linalg.norm(order[2] - next_state[5][9:12]) - np.linalg.norm(order[2] - state[3:6]))
                else:
                    reward -= (np.linalg.norm(order[1] - next_state[5][9:12]) - np.linalg.norm(order[1] - state[3:6]))

                if reward <= 0:
                    reward = -1

            if next_state[5][18] == 0:
                print("success")
                reward += 100

            prev_curriculum = curriculum
            state, curriculum = ppo_agent.convert_obs(next_state, order)

            if prev_curriculum != curriculum and curriculum == 1:
                print("deliver, next: ", curriculum)
                reward += 100
            elif prev_curriculum != curriculum and curriculum == 2:
                print("deliver, next: ", curriculum)
                reward += 100

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            current_ep_reward += reward

            # break; if the episode is over
            if done:
                if next_state[5][18] != 0:
                    reward -= 100
                break

            if time_step % 2000 == 0:
                # update PPO agent
                ppo_agent.update()

        # log in logging file
        if epi % log_freq == 0 and epi != 0:
            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{}\n'.format(epi, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if epi % print_freq == 0 and epi != 0:
            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Average Reward : {}".format(epi, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0

        # save model weights
        if epi % save_model_freq == 0 and epi != 0:
            checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, epi, run_num_pretrained)
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save_model()
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

    log_f.close()
    env.env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    main()
