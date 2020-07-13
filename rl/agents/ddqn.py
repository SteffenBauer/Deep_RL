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
    DQN agent
    
    # Arguments
      model: Keras network. Its shape must be compatible with the game.
        - input_shape: 5D tensor with shape (batch, nb_frames, height, width, channels)
        - output_shape: 2D tensor with shape (batch, nb_actions)
      mem: Replay memory object (instance of rl.memory.Memory).
        Defaults to `BasicMemory` if not specified.
      memory_size: Size of replay memory. Default to 1000.
    """

    def __init__(self, model, mem, with_target = False):
        self.model  = model
        self.with_target = with_target
        if self.with_target:
            self.target = keras.models.clone_model(model)
        self.memory = mem

        self.num_frames = self.model.input_shape[1]
        self.height     = self.model.input_shape[2]
        self.width      = self.model.input_shape[3]
        self.channels   = self.model.input_shape[4]
        self.nb_actions = self.model.output_shape[1]

        self.history = {'gamma': 0, 'epsilon': [], 'memory_fill': [],
                        'win_ratio': [], 'avg_score': [], 'max_score': []}

    def train(self, game, epochs=1, initial_epoch=1, episodes=256,
              batch_size=32, train_freq = 32, target_sync=256, gamma=0.9,
              epsilon_start=1.0, epsilon_decay=0.5, epsilon_final=0.0,
              reset_memory=False, observe=0, verbose=1, callbacks=[]):

        """
        Train the DQN agent on a game

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
          target_sync: Update the target network after these number of turns were played.
          gamma: Float between 0.0 and < 1.0. Discount factor.
          epsilon_start: Starting exploration factor between 1.0 and 0.0.
          epsilon_decay: Float between 0.0 and 1.0. Decay factor for epsilon.
          epsilon_final: Minimum value of epsilon.
          reset_memory: Boolean. Sets if the replay memory should be reset before each epoch.
            Default to `False`.
          observe: Integer. When specified, fill the replay memory with random game moves for this number of epochs.
          verbose: Integer. 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
          callbacks: List of callback objects (instance of rl.callbacks.Callback)

        # Returns
          A History dictionary, containing records of training parameters during successive epochs:
            - epsilon:      Epsilon value
            - avg_score:    Average game score
            - max_score:    Maximum reached game score
            - win_ratio:    Percentage of won games
            - memory_fill:  Records in the replay memory

        """

        self.history['gamma'] = gamma
        epsilon = epsilon_start

        if observe > 0:
            self._fill_memory(game, observe)

        if verbose > 0:
            print("{:^10s}|{:^9s}|{:^14s}|{:^9s}|{:^15s}|{:^8s}|{:^8s}".format("Epoch", "Epsilon", "Episode", "Win", "Avg/Max Score", "Memory", "Time"))

        if self.with_target:
            self.target.set_weights(self.model.get_weights())

        turn_count = 0

        for epoch in range(initial_epoch, epochs+1):
            win_count, scores, start_time = 0, [], time.time()
            if reset_memory: self.memory.reset()
            for episode in range(episodes):
                game.reset()
                for c in callbacks: 
                    c.game_start(game.get_frame())
                F = np.expand_dims(game.get_frame(), axis=0)
                S = np.repeat(F, self.num_frames, axis=0)
                current_score = 0.0
                while True:
                    action = self.act(game, S, epsilon)
                    Fn, r, game_over = game.play(action)
                    for c in callbacks: 
                        c.game_step(Fn, action, r, game_over)
                    Sn = np.append(S[1:], np.expand_dims(Fn, axis=0), axis=0)
                    self.memory.remember(S, action, r, Sn, game_over)
                    S = np.copy(Sn)
                    current_score += r
                    turn_count += 1
                    if (len(self.memory.memory) >= batch_size) and ((turn_count % train_freq) == 0):
                        result = self._train_step(gamma, batch_size)
                    if self.with_target and ((turn_count % target_sync) == 0):
                        self.target.set_weights(self.model.get_weights())
                    if game_over:
                        scores.append(current_score)
                        if game.is_won(): win_count += 1
                        for c in callbacks:
                            c.game_over()
                        break
                if verbose == 1:
                    update_progress("{0: 4d}/{1: 4d} |   {2:.2f}  | {3: 4d}".format(
                        epoch, epochs, epsilon, episode), float(episode+1)/episodes)

            win_ratio = float(win_count)/float(episodes)
            avg_score = float(sum(scores)/float(episodes))
            max_score = float(max(scores))
            memory_fill = len(self.memory.memory)
            epoch_time = time.time() - start_time
            if verbose == 2:
                print("{0: 4d}/{1: 4d} |   {2:.2f}  |    {3: 4d}    ".format(
                    epoch, epochs, epsilon, episode), end=' ')
            if verbose > 0:
                print(" | {0:>7.2%} | {1: 5.2f} /{2: 5.2f}  | {3: 6d} | {4: 5.0f}".format(
                    win_ratio, avg_score, max_score, memory_fill, epoch_time))

            self.history['epsilon'].append(epsilon)
            self.history['win_ratio'].append(win_ratio)
            self.history['avg_score'].append(avg_score)
            self.history['max_score'].append(max_score)
            self.history['memory_fill'].append(memory_fill)

            for c in callbacks: 
                c.epoch_end(
                    self.model, game.name, epoch, epsilon,
                    win_ratio, avg_score, max_score, memory_fill, epoch_time
                )

            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_final)

        return self.history

    @tf.function
    def _get_action(self, state):
        Q_values = self.model(state[np.newaxis], training=False)
        return Q_values[0] #tf.math.argmax(Q_values[0]).numpy()

    def act(self, game, state, epsilon=0.0):
        """
        Choose a game action on a given game state.

        # Arguments
          game: Game object (instance of a rl.game.Game subclass)
          state: Game state as numpy array of shape (nb_frames, height, width, channels)
          epsilon: Float between 0.0 and 1.0. Epsilon factor.
            Probability that the agent will choose a random action instead of using the DQN.

        # Returns
          The chosen game action. Integer between 0 and `game.nb_actions`.

        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.nb_actions)
        return np.argmax(self._get_action(state))

    @tf.function
    def _get_new_qvalues_target(self, gamma, actions, rewards, next_states, game_overs):
        future_rewards = self.target(next_states, training=False)
        updated_q_values = rewards + (gamma * (1.0 - game_overs) * tf.reduce_max(future_rewards, axis=1))
        mask = tf.one_hot(actions, self.nb_actions)
        return updated_q_values, mask

    @tf.function
    def _get_new_qvalues_model(self, gamma, actions, rewards, next_states, game_overs):
        future_rewards = self.model(next_states, training=False)
        updated_q_values = rewards + (gamma * (1.0 - game_overs) * tf.reduce_max(future_rewards, axis=1))
        mask = tf.one_hot(actions, self.nb_actions)
        return updated_q_values, mask

    def _train_step(self, gamma, batch_size):
        batch = self.memory.get_batch(self.model, batch_size)
        states, actions, rewards, next_states, game_overs = [
            np.array([experience[field_index] for experience in batch])
                for field_index in range(5)]
        if self.with_target:
            updated_q_values, mask = self._get_new_qvalues_target(gamma,
                tf.constant(actions), tf.constant(rewards, dtype=tf.float32),
                tf.constant(next_states, dtype=tf.float32), tf.constant(game_overs, dtype=tf.float32))
        else:
            updated_q_values, mask = self._get_new_qvalues_model(gamma,
                tf.constant(actions), tf.constant(rewards, dtype=tf.float32),
                tf.constant(next_states, dtype=tf.float32), tf.constant(game_overs, dtype=tf.float32))
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            loss = self.model.loss(updated_q_values, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def _fill_memory(self, game, episodes):
        print("Fill memory for {} episodes".format(episodes))
        for episode in range(episodes):
            game.reset()
            F = np.expand_dims(game.get_frame(), axis=0)
            S = np.repeat(F, self.num_frames, axis=0)
            while True:
                action = random.randrange(self.nb_actions)
                Fn, r, game_over = game.play(action)
                Sn = np.append(S[1:], np.expand_dims(Fn, axis=0), axis=0)
                self.memory.remember(S, action, r, Sn, game_over)
                if game_over:
                    break
            update_progress("{0: 4d}/{1: 4d} | {2: 6d} | ".
                format(episode+1, episodes, len(self.memory.memory)), float(episode+1)/episodes)
        print("")

if __name__ == '__main__':
    pass
