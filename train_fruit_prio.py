#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow.keras as keras
tf.get_logger().setLevel('ERROR')

from rl.games import fruit
from rl.agents import prioritized_dqn
from rl.memory import prioritymemory
from rl.callbacks import history

grid_size = 12
nb_frames = 1

game = fruit.Fruit(grid_size, with_penalty=False, with_poison=True)

inp = keras.layers.Input(shape=(nb_frames, grid_size, grid_size, 3))
x = keras.layers.Conv3D(16,5,padding='same',strides=1,activation='relu')(inp)
x = keras.layers.AveragePooling3D(padding='same')(x)
x = keras.layers.Conv3D(32,3,padding='same',strides=1,activation='relu')(x)
x = keras.layers.GlobalAveragePooling3D()(x)
x = keras.layers.Dense(64, activation='relu')(x)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.RMSprop(), keras.losses.LogCosh())
model.summary()

params = {
    'batch_size': 256,
    'epochs': 20,
    'episodes': 100,
    'train_freq': 4,
    'target_sync': 512,
    'epsilon_start': 0.5,
    'epsilon_decay': 0.5,
    'epsilon_final': 0.0,
    'gamma': 0.9,
    'zeta': 0.6,
    'beta': 0.4,
    'reset_memory': False,
    'observe': 100
}

rlparams = {
    'rl.memory': 'UniqMemory',
    'rl.memory_size': 65536,
    'rl.optimizer': 'RMSprop',
    'rl.with_target': True,
    'rl.nb_frames': nb_frames
}

gameparams = {
    'game.grid_size': game.grid_size,
    'game.with_poison': game.with_poison,
    'game.penalty': game.penalty,
    'game.max_turn': game.max_turn
}

memory = prioritymemory.PrioMemory(model, memory_size=rlparams['rl.memory_size'])
agent = prioritized_dqn.Agent(model, memory, with_target=rlparams['rl.with_target'])
#history = history.HistoryLog("fruit", {**params, **rlparams, **gameparams})

agent.train(game, verbose=1, callbacks=[], **params)

