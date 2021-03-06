#!/usr/bin/env python3

PROFILING = False

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow.keras as keras
tf.get_logger().setLevel('ERROR')

from rl.games import catch
from rl.agents import prioritized_dqn
from rl.memory import prioritymemory
from rl.callbacks import history

if PROFILING:
    import cProfile
    import pstats

grid_size = 16
nb_frames = 1

game = catch.Catch(grid_size, split_reward=True)

inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
x = keras.layers.Conv3D(16,3,padding='same',strides=1,activation='relu')(inp)
x = keras.layers.AveragePooling3D(padding='same')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(32, activation='relu')(x)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.RMSprop(), keras.losses.LogCosh())
model.summary()

params = {
    'batch_size': 128,
    'epochs': 20,
    'episodes': 100,
    'train_freq': 8,
    'target_sync': 512,
    'epsilon_start': 0.5,
    'epsilon_decay': 0.5,
    'epsilon_final': 0.0,
    'gamma': 0.98,
    'zeta': 0.6,
    'beta': 0.4,
    'reset_memory': False,
    'observe': 128
}

rlparams = {
    'rl.memory': 'PrioMemory',
    'rl.memory_size': 8192,
    'rl.optimizer': 'RMSprop',
    'rl.with_target': False,
    'rl.nb_frames': nb_frames
}

gameparams = {
    'game.grid_size': game.grid_size,
    'game.with_penalty': game.with_penalty,
    'game.split_reward': game.split_reward
}

memory = prioritymemory.PrioMemory(model, memory_size=rlparams['rl.memory_size'])
agent = prioritized_dqn.Agent(model, memory, with_target=rlparams['rl.with_target'])
#history = history.HistoryLog("catch", {**params, **rlparams, **gameparams})

if PROFILING:
    pr = cProfile.Profile()
    pr.enable()

agent.train(game, verbose=1, callbacks=[], **params)

if PROFILING:
    pr.disable()
    stats = pstats.Stats(pr).sort_stats('cumulative')
    stats.print_stats('Deep_RL')

