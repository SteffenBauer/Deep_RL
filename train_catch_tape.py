#!/usr/bin/env python3

PROFILING = False

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
tf.get_logger().setLevel('ERROR')

from games import catch

if PROFILING:
    import cProfile
    import pstats

grid_size = 12
nb_frames = 1

game = catch.Catch(grid_size, split_reward=True)

inp = keras.layers.Input(shape=(grid_size, grid_size, 3))
x = keras.layers.Conv2D(16,3,padding='same',strides=2,activation='relu')(inp)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(32, activation='relu')(x)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.summary()

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(game.nb_actions)
    else:
        Q_values = model(state[np.newaxis], training=False)
        return tf.math.argmax(Q_values[0]).numpy()

from collections import deque

replay_memory = deque(maxlen=1024)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(game, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done = game.play(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done

batch_size = 256
gamma = 0.95

loss_function = keras.losses.LogCosh()
optimizer = keras.optimizers.Adam(lr=1e-3)

#@tf.function
def get_new_qvalues(actions, rewards, next_states, dones):
    future_rewards = model(next_states, training=False)
    updated_q_values = rewards + (gamma * (1.0 - dones) * tf.reduce_max(future_rewards, axis=1))
    mask = tf.one_hot(actions, game.nb_actions)
    return updated_q_values, mask

def training_step(batch_size):
    states, actions, rewards, next_states, dones = sample_experiences(batch_size)
    updated_q_values, mask = get_new_qvalues(
        actions, #tf.constant(actions), 
        rewards, #tf.constant(rewards, dtype=tf.float32),
        next_states, #tf.constant(next_states, dtype=tf.float32),
        dones) #tf.constant(dones, dtype=tf.float32))
    with tf.GradientTape() as tape:
        q_values = model(states, training=True)
        q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
        loss = loss_function(updated_q_values, q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

if PROFILING:
    pr = cProfile.Profile()
    pr.enable()

wins = []

for episode in range(1000):
    game.reset()
    obs = game.get_state()
    for step in range(20):
        epsilon = max(1 - episode / 500, 0.0)
        obs, reward, done = play_one_step(game, obs, epsilon)
        if done:
            break
    wins.append(game.is_won())
    if (episode % 50) == 0:
        print("Episode: {}, Win perc: {:.2%}, eps: {:.3f}, mem: {}".format(episode, sum(wins)/float(len(wins)), epsilon, len(replay_memory)))
        wins = []

    if episode > 50:
        training_step(batch_size)

if PROFILING:
    pr.disable()
    stats = pstats.Stats(pr).sort_stats('cumulative')
    stats.print_stats()

