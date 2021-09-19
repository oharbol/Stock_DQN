import numpy as np
#import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
#from keras.optimizers import Adam
from keras.optimizer_v2.adam import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
import keras
import gym
import csv



LOAD_MODEL = None#"models/3000ep__2X64_stock_0_____0.92max____0.91avg____0.90min.model" #None #Or None

STOCK = "AAPL"

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 100_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 2  # Terminal states (end of episodes)
MODEL_NAME = '2X64_stock_1'
MIN_REWARD = -1  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 15000
# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.9995 #0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes

def convert(input):
  open_f, high_f, low_f, close_f, lips, teeth, jaw, rsi = input.split(",")
  state = [np.float(open_f), np.float(high_f), np.float(low_f), np.float(close_f), np.float(lips), np.float(teeth), np.float(jaw), np.float(rsi), 0, 0]
  # print("convert")
  # print(type(state))
  # print(type(np.array(state)))
  return np.array(state)
  

class StockEnv:
    ACTION_SPACE = 3
    HOLD_PENALTY = 0.05
    GAIN_MULT = 25
    LOSS_MULT = 100
    holding = 0
    holding_price = 0
    
    # def reset(self):
    #     with open("./data/{}_callibration.csv".format(STOCK), newline= '') as f:
    #         reader = csv.reader(f, delimiter=' ')
    #         open_f, high_f, low_f, close_f, lips, teeth, jaw, rsi = reader[0].split(",")
    #     return (float(open_f), float(high_f), float(low_f), float(close_f), float(lips), float(teeth), float(jaw), float(rsi))

    def step(self, action, line):
      # 0 = buy
      # 1 = sell
      # 2 = hold
      terminal_state = False
      gain_loss = 0
      reward = 0
      open_f, high_f, low_f, close_f, lips, teeth, jaw, rsi = line.split(",")
      if(action == 0 and self.holding == 0):
          self.holding = 1
          self.holding_price = float(open_f)
          reward = -self.HOLD_PENALTY
      elif(action == 1 and self.holding == 1):
          terminal_state = True
          gain_loss = float(open_f) - self.holding_price
          if(gain_loss > 0):
            reward = gain_loss * self.GAIN_MULT
          else:
            reward = gain_loss * -self.LOSS_MULT
          self.holding = 0
          self.holding_price = 0
      
      state = [np.float(open_f), np.float(high_f), np.float(low_f), np.float(close_f), np.float(lips), np.float(teeth), np.float(jaw), np.float(rsi), self.holding, self.holding_price]
      return np.array(state), reward, terminal_state, gain_loss


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter 

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()
        #self.model.summary()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):

        if LOAD_MODEL is not None:
          print(f"loading {LOAD_MODEL}")
          model = load_model(LOAD_MODEL)
          print(f"Model {LOAD_MODEL} loaded!")

        else:
          model = Sequential()
          model.add(Dense(64, input_shape=(10,))) #Change activation space to be (8) ohlc, lips, teeth, jaw, rsi
          model.add(Activation('relu'))


          model.add(Dense(128))
          model.add(Activation('relu'))

          model.add(Dense(128))
          model.add(Activation('relu'))

          #buy, hold, sell
          model.add(Dense(3, activation='linear'))
          model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # print("mini")
        # print(minibatch)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch], dtype=np.object)
    
        #Need to do this to prevent errors FUCK!!
        current_states = np.asarray(current_states).astype(np.float)
        # again = np.array([transition[0] for transition in minibatch])
        # print("again")
        # print(again)

        # Normalize data
        # for index, i in enumerate(current_states):
        #   current_states[index][0] = i[0] / MAX_POSX
        #   current_states[index][1] = i[1] / MAX_POSY
        #   current_states[index][2] = i[2] / MAX_VELX
        #   current_states[index][3] = i[3] / MAX_VELY
        #   current_states[index][4] = i[4] / MAX_ANGLE
        #   current_states[index][5] = i[5] / MAX_ANGVEL

        current_qs_list = self.model.predict(current_states)
        #print(current_qs_list)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        # Normalize data
        # for index, i in enumerate(new_current_states):
        #   new_current_states[index][0] = i[0] / MAX_POSX
        #   new_current_states[index][1] = i[1] / MAX_POSY
        #   new_current_states[index][2] = i[2] / MAX_VELX
        #   new_current_states[index][3] = i[3] / MAX_VELY
        #   new_current_states[index][4] = i[4] / MAX_ANGLE
        #   new_current_states[index][5] = i[5] / MAX_ANGVEL

        #print(new_current_states)

        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index]) 
                new_q = reward + DISCOUNT * max_future_q #current_qs_list[index][action] + reward + DISCOUNT * (max_future_q - current_qs_list[index][action])
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            #print(current_qs)
            #print(current_qs)
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)

            y.append(current_qs)

        # Normalize X data
        # for index, i in enumerate(X):
        #   X[index][0] = i[0] / MAX_POSX
        #   X[index][1] = i[1] / MAX_POSY
        #   X[index][2] = i[2] / MAX_VELX
        #   X[index][3] = i[3] / MAX_VELY
        #   X[index][4] = i[4] / MAX_ANGLE
        #   X[index][5] = i[5] / MAX_ANGVEL
        #print(y)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        #print("HERE3")
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

env = StockEnv()
#state:
#Open, High, Low, Close, Lips, Teeth, Jaw, Ris
agent = DQNAgent()
ep_rewards = [0]
ep_gain = [0]

# open csv of normalized data
state_file = open("./data/{}_norm.csv".format(STOCK))
current_state = convert(next(state_file)) #env.reset()

days = 0
step = 1
profit = 0
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    episode_gain = 0

    # Reset flag and start iterating until episode ends
    done = False
    while not done:#step != 390:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
            
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE)

        #get the next ohlc, alligator, and rsi
        line = next(state_file)
        #add gain_loss variable
        new_state, reward, done, gain_loss = env.step(action, line)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward
        episode_gain += gain_loss
        profit += gain_loss

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        current_state = new_state
        step += 1
        if(step == 390):
          step = 1
          days += 1
        #print(step)

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    ep_gain.append(episode_gain)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        average_gain = sum(ep_gain[-AGGREGATE_STATS_EVERY:])/len(ep_gain[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, gain_loss=average_gain, profit=profit)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{episode}ep__{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
print()
print(days)
state_file.close()