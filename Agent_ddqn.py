# submission을 위한 Agent 파일 입니다.
# policy(), save_model(), load_model()의 arguments와 return은 변경하실 수 없습니다.
# 그 외에 자유롭게 코드를 작성 해주시기 바랍니다.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from Drone_gym import Drone_gym
import random
import os

## 이미지 관련 코드 
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray

class Q_network(nn.Module):
    def __init__(self):
        super(Q_network, self).__init__()

        self.vector_fc = nn.Linear(26, 64) # embedding vector observation
        
        # self.image_cnn = nn.Sequential(
        #     nn.Conv2d(4,32,kernel_size=16,stride=8),
        #     nn.ReLU()
        #     )
        # self.image_fc = nn.Linear(1568,128) ## embedding image observation

        self.fc_action1 = nn.Linear(64, 64)
        self.fc_action2 = nn.Linear(64, 7)

    def forward(self,x):
        
        vector_obs = x
        batch = vector_obs.size(0)

        vector_obs = torch.relu(self.vector_fc(vector_obs))

        # front_obs = self.image_fc(self.image_cnn(front_obs).view(batch,-1))
        # right_obs = self.image_fc(self.image_cnn(right_obs).view(batch,-1))
        # rear_obs = self.image_fc(self.image_cnn(rear_obs).view(batch,-1))
        # left_obs = self.image_fc(self.image_cnn(left_obs).view(batch,-1))
        # btn_obs = self.image_fc(self.image_cnn(btn_obs).view(batch,-1))

        # image_obs = (front_obs + right_obs + rear_obs + left_obs + btn_obs)
        #
        # x = torch.cat((vector_obs,image_obs),-1)

        action_logit = torch.relu(self.fc_action1(vector_obs))
        Q_values  = self.fc_action2(action_logit)

        return Q_values 
     
class Agent:
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.policy_net = Q_network().to(device)
        self.Q_target_net = Q_network().to(device)
        self.learning_rate = 0.0003

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.Q_target_net.load_state_dict(self.policy_net.state_dict()) ## Q의 뉴럴넷 파라미터를 target에 복사
        
        self.epsilon = 1 ## epsilon 1 부터 줄어들어감.
        self.epsilon_decay = 0.00001 ## episilon 감쇠 값.
        
        self.device = device
        self.data_buffer = deque(maxlen=10240)

        self.init_check = 0
        self.cur_state = None
        
    def epsilon_greedy(self, Q_values):
        if np.random.rand() <= self.epsilon: ## 0~1의 균일분포 표준정규분포 난수를 생성. 정해준 epsilon 값보다 작은 random 값이 나오면 
            action = random.randrange(6) ## action을 random하게 선택합니다.
            return action
        
        else: ## epsilon 값보다 크면, 학습된 Q_player NN 에서 얻어진 Q_values 값중 가장 큰 action을 선택합니다.
            return Q_values.argmax().item()
        
    def policy(self, obs): 
        """Policy function p(a|s), Select three actions.

        Args:
            obs: all observations. consist of 6 observations.
                   (front image observation, right image observation, 
                   rear image observation, left image observation, 
                   bottom observation, vector observation)

        Return:
            3 continous actions vector (range -1 ~ 1) from policy.
            ex) [-1~1, -1~1, -1~1]
        """
        if self.init_check == 0:
            self.cur_obs = self.init_obs(obs)
            self.init_check += 1
        else:
            self.cur_obs = self.accumulated_all_obs(self.cur_obs, obs)
        Q_values = self.policy_net(self.torch_obs( self.cur_obs))
        action = self.epsilon_greedy(Q_values)
     
        return self.convert_action(action) ## inference를 위한 (x, y, z) 위치 action, 각각 0~1 사이의 continuous value.

    def train_policy(self, obs): 
        """Policy function p(a|s), Select three actions.

        Args:
            obs: all observations. consist of 6 observations.
                   (front image observation, right image observation, 
                   rear image observation, left image observation, 
                   bottom observation, vector observation)

        Return:
            3 continous actions vector (range -1 ~ 1) from policy.
            ex) [-1~1, -1~1, -1~1]
        """
        Q_values = self.policy_net(obs)
        action = self.epsilon_greedy(Q_values)
     
        return self.convert_action(action), action
    
    def save_model(self):
        torch.save(self.policy_net.state_dict(), './best_model/best_model.pkl')
        return None

    def load_model(self):
        self.policy_net.load_state_dict(torch.load('./best_model/best_model.pkl', map_location=self.device))
        return None

    def store_trajectory(self, traj):
        """
           store data
        """
        self.data_buffer.append(traj)

    def re_scale_frame(self, obs):
        """
        change rgb to gray.
        """
        return resize(rgb2gray(obs),(64,64))

    def init_image_obs(self, obs):
        """
        set initial observation s_0, stacked 4 s_0 frames.
        """
        obs = self.re_scale_frame(obs)
        frame_obs = [obs for _ in range(4)]
        frame_obs = np.stack(frame_obs, axis=0)

        return frame_obs

    def init_obs(self, obs, order):
        """
           set initial observation
        """

        # front_obs = self.init_image_obs(obs[0])
        # right_obs = self.init_image_obs(obs[1])
        # rear_obs = self.init_image_obs(obs[2])
        # left_obs = self.init_image_obs(obs[3])
        # btn_obs = self.init_image_obs(obs[4])
        vector_obs = []
        target_obs = order[0] # 3
        robot_obs = obs[5][9:18] # 9
        lidar_obs = obs[5][19:43] #24
        lidar_obs = np.append(lidar_obs, obs[5][91:95]) #4
        vector_obs = np.append(vector_obs, target_obs)
        vector_obs = np.append(vector_obs, robot_obs)
        for i in range(14): # len(lidar_obs)/2 = 14
            if lidar_obs[2*i] == 1:
                vector_obs = np.append(vector_obs, 0)
            else:
                vector_obs = np.append(vector_obs, lidar_obs[2*i + 1])
        
        return vector_obs

    def torch_obs(self, obs):
        """
            convert to torch tensor
        """
        # front_obs = torch.Tensor(obs[0]).unsqueeze(0).to(self.device)
        # right_obs = torch.Tensor(obs[1]).unsqueeze(0).to(self.device)
        # rear_obs = torch.Tensor(obs[2]).unsqueeze(0).to(self.device)
        # left_obs = torch.Tensor(obs[3]).unsqueeze(0).to(self.device)
        # btn_obs = torch.Tensor(obs[4]).unsqueeze(0).to(self.device)
        vector_obs = torch.Tensor(obs).to(self.device)
  
        return vector_obs

    def accumulated_image_obs(self, obs, new_frame):
        """
            accumulated image observation.
        """
        temp_obs = obs[1:,:,:]
        new_frame = np.expand_dims(self.re_scale_frame(new_frame), axis=0)
        frame_obs = np.concatenate((temp_obs, new_frame),axis=0)
    
        return frame_obs

    def accumulated_all_obs(self, obs, next_obs, order):
        """
            accumulated all observation.
        """
        # front_obs = self.accumulated_image_obs(obs[0], next_obs[0])
        # right_obs = self.accumulated_image_obs(obs[1], next_obs[1])
        # rear_obs = self.accumulated_image_obs(obs[2], next_obs[2])
        # left_obs = self.accumulated_image_obs(obs[3], next_obs[3])
        # btn_obs = self.accumulated_image_obs(obs[4], next_obs[4])
        vector_obs = []
        if next_obs[5][18] == 1.0:
            target_obs = order[0] #3
            curriculum = 0
        elif next_obs[5][18] < 0.4:
            target_obs = order[2] #3
            curriculum = 2
        else:
            target_obs = order[1] #3
            curriculum = 1
        robot_obs = next_obs[5][9:18] # 9
        lidar_obs = next_obs[5][19:43] #24
        lidar_obs = np.append(lidar_obs, next_obs[5][91:95]) #4
        vector_obs = np.append(vector_obs, target_obs)
        vector_obs = np.append(vector_obs, robot_obs)
        for i in range(14): # len(lidar_obs)/2 = 38
            if lidar_obs[2*i] == 1:
                vector_obs = np.append(vector_obs, 0)
            else:
                vector_obs = np.append(vector_obs, lidar_obs[2*i + 1])
     
        return vector_obs, curriculum

    def convert_action(self, action):
        if action == 0:
            return [1, 0, 0]
        elif action == 1:
            return [0, 1,  0]
        elif action == 2:
            return [0, 0,  1]
        elif action == 3:
            return [-1,  0, 0]
        elif action == 4:
            return [0,  -1,  0]
        elif action == 5:
            return [0,  0,  -0.1]
        else:
            return [0, 0, 0]

    def batch_torch_obs(self, obs):
        # front_obs = torch.Tensor(np.stack([s[0] for s in obs], axis=0)).to(self.device)
        # right_obs = torch.Tensor(np.stack([s[1] for s in obs], axis=0)).to(self.device)
        # rear_obs = torch.Tensor(np.stack([s[2] for s in obs], axis=0)).to(self.device)
        # left_obs = torch.Tensor(np.stack([s[3] for s in obs], axis=0)).to(self.device)
        # btn_obs = torch.Tensor(np.stack([s[4] for s in obs], axis=0)).to(self.device)
        vector_obs = torch.Tensor(np.stack([s for s in obs], axis=0)).to(self.device)
        
        return vector_obs

    def update_target(self):
        self.Q_target_net.load_state_dict(self.policy_net.state_dict()) ## Q_player NN에 학습된 weight를 그대로 Q_target에 복사함.

    def train(self):
        
        gamma = 0.99

        self.epsilon -= self.epsilon_decay 
        self.epsilon = max(self.epsilon, 0.05) 
        random_mini_batch = random.sample(self.data_buffer, 1024) ## batch_size 만큼 random sampling.
        
        # data 분배
        obs_list, action_list, reward_list, next_obs_list, mask_list = [], [], [], [], []
        
        for all_obs in random_mini_batch:
            s, a, r, next_s, mask = all_obs
            obs_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_obs_list.append(next_s)
            mask_list.append(mask)
        
        # tensor.
        obses = self.batch_torch_obs(obs_list)
        actions = torch.LongTensor(action_list).unsqueeze(1).to(self.device)
        rewards = torch.Tensor(reward_list).to(self.device)
        next_obses = self.batch_torch_obs(next_obs_list)
        masks = torch.Tensor(mask_list).to(self.device)

        # get Q-value
        Q_values = self.policy_net(obses)
        q_value = Q_values.gather(1, actions).view(-1)
        
        # get target
        target_q_value = self.Q_target_net(next_obses).max(1)[0] 
                                                                
        Y = rewards + masks * gamma * target_q_value

        # loss 정의 
        MSE = torch.nn.MSELoss() 
        loss = MSE(q_value, Y.detach())
        
        # backward 시작!
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
            port=12000,
            filename='../RL_Drone/drone.x86_64')
    '''
    해상도 변경을 원할 경우, width, height 값 조절.
    env = Drone_gym(
            time_scale=1.0,
            port=11000,
            width=84, height=84, filename='../RL_Drone/DroneDelivery.exe')
    '''
    score = 0
    reward_score = 0
    episode_step = 0
    train_step = 0
    episode = 999999999999
    print_freq = 50

    initial_exploration = 2000
    update_target = 2000
    
    agent = Agent()

    #### log files for multiple runs are NOT overwritten

    env_name="drone"
    log_dir = "DDQN_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### create new log file for each run
    #### get number of log files in log directory
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    log_f_name = log_dir + '/DDQN_' + env_name + "_log_" + str(run_num) + ".csv"

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,reward\n')

    try:
        agent.load_model() # user의 모델 불러오기. 경로는 best_model 폴더.
        print("Succeed load User's model")
    except:
        print("Fail load User's model'")
        raise

    for epi in range(episode):
        obs = env.reset()
        order = agent.best_order(obs[5][9:12], obs[5][:3], obs[5][3:6], obs[5][6:9])
        prev_curriculum = 0
        obs = agent.init_obs(obs, order)

        while True:
            action, dis_action = agent.train_policy(agent.torch_obs(obs))
            next_obs, reward, done = env.step(action)

            next_obs, curriculum = agent.accumulated_all_obs(obs, next_obs, order)
            mask = 0 if done else 1
            
            score += reward
            # reward = np.clip(reward, -2, 2) # reward clipping 사용 자유.
            reward_score += reward
            agent.store_trajectory((obs, dis_action, reward, next_obs,  mask))
            obs = next_obs
            if train_step > initial_exploration: ## 초기 exploration step을 넘어서면 학습 시작.
                agent.train()
                if train_step % update_target == 0: ## 일정 step마다 target network업데이트
                    agent.update_target()

            episode_step += 1
            train_step += 1

            if prev_curriculum != curriculum:
                print("deliver, next: ", curriculum)
            prev_curriculum = curriculum

            if done:
                break

        if epi % print_freq == 0 and epi != 0:
            print('episode: ', epi, ' True_score: ', score, ' everage score: ', reward_score/episode_step)

            log_avg_reward = round(reward_score/episode_step, 4)
            log_f.write('{},{}\n'.format(epi, log_avg_reward))
            log_f.flush()
        if epi % 1000 == 0 and epi != 0:
            agent.save_model()  # 1000 step마다 model 저장.
        score = 0
        reward_score = 0
        episode_step = 0

    log_f.close()

if __name__ == '__main__':
    main()
