#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow.keras as keras
tf.get_logger().setLevel('ERROR')

from rl.games import tromis
from rl.agents import dqn
from rl.memory import uniqmemory
from rl.callbacks import history

width, height = 7, 10
nb_frames = 1

game = tromis.Tromis(width, height, max_turn=512)

inp = keras.layers.Input(shape=(nb_frames, height, width, 3))
x = keras.layers.Conv3D(32,5,padding='same',strides=1,activation='relu')(inp)
x = keras.layers.AveragePooling3D(padding='same')(x)
x = keras.layers.Conv3D(64,3,padding='same',strides=1,activation='relu')(x)
x = keras.layers.GlobalAveragePooling3D()(x)
x = keras.layers.Dense(128, activation='relu')(x)
act = keras.layers.Dense(game.nb_actions, activation='linear')(x)

model = keras.models.Model(inputs=inp, outputs=act)
model.compile(keras.optimizers.RMSprop(), keras.losses.LogCosh())
model.summary()

params = {
    'batch_size': 256,
    'epochs': 200,
    'episodes': 100,
    'train_freq': 32,
    'target_sync': 512,
    'epsilon_start': 0.5,
    'epsilon_decay': 0.75,
    'epsilon_final': 0.0,
    'gamma': 0.92,
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
    'game.width': game.width,
    'game.height': game.height,
    'game.max_turn': game.max_turn
}

memory = uniqmemory.UniqMemory(memory_size=rlparams['rl.memory_size'])
agent = dqn.Agent(model, memory, with_target=rlparams['rl.with_target'])
#history = history.HistoryLog("tromis", {**params, **rlparams, **gameparams})

agent.train(game, verbose=1, callbacks=[], **params)
