#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import sys
import random
import time

def update_progress(msg, progress):
    text = "\r{0} {1:>7.2%}".format(msg, progress)
    sys.stdout.write(text)
    sys.stdout.flush()

class Agent(object):
    """
    Policy gradient agent
    
    # Arguments
      model: Keras network. Its shape must be compatible with the game.
        - input_shape: 5D tensor with shape (batch, nb_frames, height, width, channels)
        - output_shape: 2D tensor with shape (batch, nb_actions)
    """

    def __init__(self, model):
        self.model  = model

        self.num_frames = self.model.input_shape[1]
        self.height     = self.model.input_shape[2]
        self.width      = self.model.input_shape[3]
        self.channels   = self.model.input_shape[4]
        self.nb_actions = self.model.output_shape[1]

        self.history = {'gamma': 0, 'win_ratio': [], 'avg_score': [], 'max_score': []}

    def train(self, game, epochs=1, initial_epoch=1, episodes=256,
              batch_size=32, train_freq = 32, gamma=0.9,
              observe=0, verbose=1, callbacks=[]):

        """
        Train the PG agent on a game

        # Arguments
          game: Game object (instance of a qlearn.game.Game subclass)
          epochs: Integer. Number of epochs to train the model.
            When unspecified, the network is trained over one epoch.
          initial_epoch: Integer. Value to start epoch counting.
            If unspecified, `initial_epoch` will default to 1.
            This argument is useful when continuing network training.
          episodes: Integer. Number of game episodes to play during one epoch.
          batch_size: Integer. Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
          train_freq: Integer. Train the Q-Network after these number of turns were played.
          gamma: Float between 0.0 and < 1.0. Discount factor.
          observe: Integer. When specified, fill the replay memory with random game moves for this number of epochs.
          verbose: Integer. 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
          callbacks: List of callback objects (instance of rl.callbacks.Callback)

        # Returns
          A History dictionary, containing records of training parameters during successive epochs:
            - avg_score:    Average game score
            - max_score:    Maximum reached game score
            - win_ratio:    Percentage of won games
        """
        self.history['gamma'] = self.gamma = gamma
        if verbose > 0:
            print("{:^10s}|{:^14s}|{:^9s}|{:^15s}|{:^8s}".format("Epoch", "Episode", "Win", "Avg/Max Score", "Time"))

        for epoch in range(initial_epoch, epochs+1):
            win_count, scores, start_time = 0, [], time.time()
            for episode in range(episodes):
                game.reset()
                transitions = [] # list of state, action, rewards
                for c in callbacks: 
                    c.game_start(game.get_frame())
                F = np.expand_dims(game.get_frame(), axis=0)
                S = np.repeat(F, self.num_frames, axis=0)
                current_score = 0.0
                while True:
                    action = self.act(game, S)
                    Fn, r, game_over = game.play(action)
                    transitions.append((S, action, r))
                    for c in callbacks: 
                        c.game_step(Fn, action, r, game_over)
                    Sn = np.append(S[1:], np.expand_dims(Fn, axis=0), axis=0)
                    S = np.copy(Sn)
                    current_score += r
                    turn_count += 1
                    if r != 0.0:
                        self.train(transitions)
                        transitions = []
                    if game_over:
                        scores.append(current_score)
                        if game.is_won(): win_count += 1
                        for c in callbacks:
                            c.game_over()
                        break
                if verbose == 1:
                    update_progress("{0: 4d}/{1: 4d} | {2: 4d}".format(
                        epoch, epochs, episode), float(episode+1)/episodes)



    @tf.function
    def act(self, game, state):
        act_prob = self.model.(state[np.newaxis], training=False)
        action = tf.random.categorical(tf.math.log([act_prob], 1))
        return action[0][0].numpy()

    def loss_fn(y_true, y_pred):
        return -1.0 * tf.math.reduce_sum(y_true * tf.math.log(y_pred))

    def train(self, transitions):
        ep_len = len(transitions)
        discounted_rewards = np.zeros((ep_len, game.nb_actions))
        train_states = []
        for i in range(ep_len):
            discount = 1.0
            future_reward = 0.0
            # discount rewards
            for i2 in range(i, ep_len):
                future_reward += transitions[i2][2] * discount
                discount = discount * self.gamma
            discounted_rewards[i][transitions[i][1]] = future_reward
            train_states.append(transitions[i][0])
        train_states = np.asarray(train_states)
        # Backpropagate model with preds & discounted_rewards here
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            masked_predictions = 
            loss = self.model.loss(masked_predictions, discounted_rewards)
        
        loss = self.model.train_on_batch(train_states, discounted_rewards)
        loss_stats.append(loss)

if __name__ == '__main__':
    pass
