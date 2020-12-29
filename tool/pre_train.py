import sys
import argparse
import pickle
import pathlib
import random
import math
import numpy as np
import time
import itertools
from matplotlib import pyplot as plt

from stem_lib import learning_utils

def main(args):

    sample_dict_path = pathlib.Path(args.sample_dict)

    with (sample_dict_path).open('rb') as file:
        sample_dict = pickle.load(file)

    

    # print(np.linalg.norm(np.array(sample_dict['inflating'][0][0]) - np.array(sample_dict['inflating'][100][0])))


    print_each_length(sample_dict)

    head_state = next(iter(sample_dict))
    frame_size = len(sample_dict[head_state][0])
    segment_size = len(sample_dict[head_state][0][0])
    print(f'frame_size: {frame_size}')
    print(f'segment_size: {segment_size}')


    # slice valid range
    # and convert frame into np.array
    sample_size = 200
    for name in sample_dict:
        sample_dict[name] = np.array(sample_dict[name][0:sample_size])

    print_each_length(sample_dict)

    # suffle
    for name in sample_dict:
        random.shuffle(sample_dict[name])

    print_each_length(sample_dict)


    n_states = len(sample_dict)
    shape = [math.ceil(n_states / 2), 2]
    fig, subplots = plt.subplots(*shape)
    for plot_idx, name, frames in zip(itertools.product(*[range(s) for s in shape]), sample_dict.keys(), sample_dict.values()):
        subplot = subplots[plot_idx[0]][plot_idx[1]]
        subplot.set_title(name)
        subplot.plot(frames[0])
        
    plt.show()

    # divide train and test 
    train_sample_dict = {name: frames[0:math.floor(sample_size/2)] for name, frames in sample_dict.items()}
    test_sample_dict = {name: frames[math.floor(sample_size/2) : sample_size] for name, frames in sample_dict.items()}

    print('train_sample_dict: ')
    print_each_length(train_sample_dict)

    print('test_sample_dict: ')
    print_each_length(test_sample_dict)

    # devide train data into each batch
    batches = []
    batch_size = 20

    for idx in range(0, min([ len(frames) for frames in train_sample_dict.values()]) , batch_size):
        batch = {}
        for name, frames in train_sample_dict.items():
            batch[name] = frames[idx: idx + batch_size]
            
        batches.append(batch)


    # # setup train data
    # train_data = []
    # for name, frames in train_sample_dict.items():
    #     train_data.extend([{'name': name, 'frame': frame} for frame in frames])

    # random.shuffle(train_data)

    # print(len(train_data))
    # print(train_data[0])

    
    for batch in batches:
        for name, frames in batch.items():
            print(name, len(frames))

    # train
    model = learning_utils.make_model(frame_size, segment_size)

    # frame1 = np.random.rand(1, frame_size, segment_size)
    # frame2 = np.random.rand(1, frame_size, segment_size)
    # frame3 = np.random.rand(1, frame_size, segment_size)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(221, title='ax1')
    # ax2 = fig.add_subplot(222, title='ax2')
    # ax3 = fig.add_subplot(223, title='ax3')

    # print(np.shape(model(frame1)))
    # ax1.plot(model(frame1)[0])
    # ax2.plot(model(frame2)[0])
    # ax3.plot(model(frame3)[0])

    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    print('evaluate')
    evaluate(model, test_sample_dict)

    for batch in batches:

        print('train')
        for step in range(3):
            train_per_batch(batch, model)

        print('evaluate')
        evaluate(model, test_sample_dict)

    # test

    # print(sample_dict['non'])
    # sample_dict['non'] = sample_dict['non'][0:25]

    # each_length = {name: len(frames) for name, frames in sample_dict.items()}
    # print(f'each_length: \n{each_length}')
    # print(sample_dict['non'])

    # with (sample_dict_path).open('wb') as file:
    #     pickle.dump(sample_dict, file)
    # print(sample_dict)

def train_per_batch(batch, model):
    embedding_dict = {}
    for name, frames in batch.items():
        # embedding_dict[name], _ = model(np.array(frames))
        embedding_dict[name] = model(np.array(frames))
    
    input_frames = []
    target_embeddings = []
    for anchor_name, frames in batch.items():
        for anchor_idx, anchor_frame in enumerate(frames):
            anchor_embedding = embedding_dict[name][anchor_idx]
            positive_embeddings, _ = get_embeddings(embedding_dict, lambda name, idx: name == anchor_name and idx != anchor_idx )
            negative_embeddings, _ = get_embeddings(embedding_dict, lambda name, idx: name != anchor_name)

            triplets = learning_utils.select_triplets(anchor_embedding, positive_embeddings, negative_embeddings)

            for anchor_embedding, positive_embedding, negative_embedding in triplets:
                input_frames.append(anchor_frame)
                target_embeddings.append([positive_embedding, negative_embedding])


            # print(anchor_name, len(positive_embeddings), len(negative_embeddings))

    input_frames = np.array(input_frames)
    target_embeddings = np.array(target_embeddings)
    print(len(input_frames), len(target_embeddings))

    model.fit(input_frames, target_embeddings, batch_size=32)

def evaluate(model, test_sample_dict):
    embedding_dict = {}
    for name, frames in test_sample_dict.items():
        # embedding_dict[name], _ = model(np.array(frames))
        embedding_dict[name] = model(np.array(frames))
    
    for anchor_name, frames in test_sample_dict.items():
        for anchor_idx, anchor_frame in enumerate(frames):
            print(anchor_name)
            anchor_embedding = embedding_dict[name][anchor_idx]
            positive_embeddings, pos_emb_locs = get_embeddings(embedding_dict, lambda name, idx: name == anchor_name and idx != anchor_idx )
            negative_embeddings, neg_emb_locs = get_embeddings(embedding_dict, lambda name, idx: name != anchor_name)

            anchor_embedding = np.array(anchor_embedding)
            positive_embeddings = np.array(positive_embeddings)
            negative_embeddings = np.array(negative_embeddings)

            print(anchor_name, anchor_idx)
            print(pos_emb_locs[0]['name'], pos_emb_locs[0]['index'])
            print(neg_emb_locs[0]['name'], neg_emb_locs[0]['index'])

            # print(anchor_embedding[0], anchor_embedding[1], anchor_embedding[2])

            # import ipdb; ipdb.set_trace()
            
            fig, subplots = plt.subplots(2, 2)
            subplots[0][0].set_title('all')
            subplots[0][0].plot(negative_embeddings[0], label='negative')
            subplots[0][0].plot(positive_embeddings[0], label='positive')
            subplots[0][0].plot(anchor_embedding, label='anchor')

            subplots[0][1].set_title(f'anchor ({anchor_name})')
            subplots[0][1].plot(anchor_embedding, label='anchor')

            subplots[1][0].set_title('positive')
            subplots[1][0].plot(positive_embeddings[0], label='positive')

            subplots[1][1].set_title('negative')
            subplots[1][1].plot(negative_embeddings[0], label='negative')
            
            
            # ax1.plot(anchor_frame)
            # ax1.plot(test_sample_dict[pos_emb_locs[0]['name']][pos_emb_locs[0]['index']])
            # ax1.plot(test_sample_dict[neg_emb_locs[0]['name']][neg_emb_locs[0]['index']])

            # ax2.plot(anchor_embedding)
            # ax3.plot(positive_embeddings[0])
            # ax4.plot(negative_embeddings[0])

            fig.legend()
            # plt.tight_layout()

            print('p: ', np.linalg.norm(anchor_embedding - positive_embeddings), '; n: ', np.linalg.norm(anchor_embedding - negative_embeddings))
            print(np.linalg.norm(anchor_embedding, ord=2))
            # print(np.shape(anchor_embedding))
            break # next state
    
    plt.show()

def get_embeddings(embedding_dict, filter_func):
    filtered_embs = []
    locations = []
    for name, embs in embedding_dict.items():
        for idx, emb in enumerate(embs):
            if filter_func(name, idx):
                locations.append({'name': name, 'index': idx})
                filtered_embs.append(emb)
    return filtered_embs, locations


def print_each_length(d):
    each_length = {key: len(value) for key, value in d.items()}
    print(f'each_length: \n{each_length}')



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sample-dict',
        help="Path to the sample_dict.pkl file.",
        default='.stem/samples/sample_dict.pkl'
    )

    parser.add_argument(
        '--valid-range-size',
        default=-1
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
