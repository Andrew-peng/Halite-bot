import datetime
import os
import sys

import numpy as np
import json
import itertools

from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

REPLAY_FOLDER = sys.argv[1]
training_input = []
training_target = []

VISIBLE_DISTANCE = 4
input_dim=4*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)
np.random.seed(0) # for reproducibility

model = Sequential([Dense(512, input_dim=input_dim),
                    LeakyReLU(),
                    Dense(512),
                    LeakyReLU(),
                    Dense(512),
                    LeakyReLU(),
                    Dense(5, activation='softmax')])
model.compile('nadam','categorical_crossentropy', metrics=['accuracy'])
def rotate_moves(moves, num):
    for x in range(len(moves)):
        for y in range(len(moves[x])):
            for z in range(len(moves[x][y])):
                if moves[x][y][z] == 0:
                    continue
                else:
                    moves[x][y][z] = (moves[x][y][z] + num) % 4
                    if moves[x][y][z] == 0:
                        moves[x][y][z] = 4
    return moves
def stack_to_input(stack, position):
    return np.take(np.take(stack,
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],axis=1,mode='wrap'),
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],axis=2,mode='wrap').flatten()

size = len(os.listdir(REPLAY_FOLDER))
for index, replay_name in enumerate(os.listdir(REPLAY_FOLDER)):
    if replay_name[-4:]!='.hlt':continue
    print('Loading {} ({}/{})'.format(replay_name, index, size))
    replay = json.load(open('{}/{}'.format(REPLAY_FOLDER,replay_name)))
    for i in range(4):
        ks=np.array(replay['frames'])
        a = np.rot90(ks[0], i, axes=(1,0)).shape
        frames = np.empty(shape = (replay['num_frames'], a[0], a[1], 2))
        for s in range(replay['num_frames']):
            frames[s] = np.rot90(ks[s], i, axes=(1,0))
        player=frames[:,:,:,0]
        players,counts = np.unique(player[-1],return_counts=True)
        target_id = players[counts.argmax()]
        if target_id == 0: continue

        prod = np.repeat(np.rot90(np.array(replay['productions']), i, axes=(1,0))[np.newaxis],replay['num_frames'],axis=0)
        strength = frames[:,:,:,1]

        translated = np.array(replay['moves'])
        newarr = np.empty(shape = (replay['num_frames'] - 1, a[0], a[1]))
        for s in range(replay['num_frames'] - 1):
            newarr[s] = np.rot90(translated[s], i, axes=(1,0))
        newarr = rotate_moves(newarr, i)
        moves = (np.arange(5) == newarr[:,:,:,None]).astype(int)[:128]
        stacks = np.array([player==target_id,(player!=target_id) & (player!=0),prod/20,strength/255])
        stacks = stacks.transpose(1,0,2,3)[:len(moves)].astype(np.float32)

        position_indices = stacks[:,0].nonzero()
        sampling_rate = 1/stacks[:,0].mean(axis=(1,2))[position_indices[0]]
        sampling_rate *= moves[position_indices].dot(np.array([1,10,10,10,10])) # weight moves 10 times higher than still
        sampling_rate /= sampling_rate.sum()
        sample_indices = np.transpose(position_indices)[np.random.choice(np.arange(len(sampling_rate)),
                                                                        min(len(sampling_rate),2048),p=sampling_rate,replace=False)]

        replay_input = np.array([stack_to_input(stacks[i],[j,k]) for i,j,k in sample_indices])
        replay_target = moves[tuple(sample_indices.T)]

        training_input.append(replay_input.astype(np.float32))
        training_target.append(replay_target.astype(np.float32))

now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
training_input = np.concatenate(training_input,axis=0)
training_target = np.concatenate(training_target,axis=0)
indices = np.arange(len(training_input))
np.random.shuffle(indices) #shuffle training samples
training_input = training_input[indices]
training_target = training_target[indices]

model.fit(training_input,training_target,validation_split=0.2,
          callbacks=[EarlyStopping(patience=15, min_delta=0.00001),
                     ModelCheckpoint('model7.h5',verbose=1,save_best_only=True),
                     tensorboard],
          batch_size=100, nb_epoch=1000)

model = load_model('model7.h5')

still_mask = training_target[:,0].astype(bool)
print('STILL accuracy:',model.evaluate(training_input[still_mask],training_target[still_mask],verbose=0)[1])
print('MOVE accuracy:',model.evaluate(training_input[~still_mask],training_target[~still_mask],verbose=0)[1])
