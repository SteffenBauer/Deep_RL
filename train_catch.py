#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow.keras as keras
tf.get_logger().setLevel('ERROR')

from rl.games import catch
from rl.agents import ddqn
from rl.memory import uniqmemory
from rl.callbacks import history

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
    'batch_size': 256,
    'epochs': 20,
    'episodes': 100,
    'train_freq': 8,
    'target_sync': 512,
    'epsilon_start': 0.5,
    'epsilon_decay': 0.5,
    'epsilon_final': 0.0,
    'gamma': 0.98,
    'reset_memory': False,
    'observe': 128
}

rlparams = {
    'rl.memory': 'UniqMemory',
    'rl.memory_size': 8192,
    'rl.optimizer': 'RMSprop',
    'rl.with_target': True,
    'rl.nb_frames': nb_frames
}

gameparams = {
    'game.grid_size': game.grid_size,
    'game.with_penalty': game.with_penalty,
    'game.split_reward': game.split_reward
}

memory = uniqmemory.UniqMemory(memory_size=rlparams['rl.memory_size'])
agent = ddqn.Agent(model, memory, with_target=rlparams['rl.with_target'])
#history = history.HistoryLog("catch", {**params, **rlparams, **gameparams})

agent.train(game, verbose=1, callbacks=[], **params)

