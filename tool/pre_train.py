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

from stem_lib import stem_utils
from stem_lib import learning_utils
from stem_lib.stdlib import file_utils as std_file_utils
from stem_lib.stdlib.logging import TeeLogger


context = {
    'output_dir': None,
    'epoch_step': 0,
    'batch_step': 0
}

def main(args):

    context['output_dir'] = stem_utils.STEM_CONSTANTS.STEM_LOG_DIR / 'pre_train' / std_file_utils.get_unique_log_dir()
    context['output_dir'].mkdir(parents=True)

    sys.stdout = TeeLogger(context['output_dir'] / 'stdout.log')

    print(f'output_dir\t: {context["output_dir"]}')


    sample_dict_path = pathlib.Path(args.sample_dict)
    print(f"sample_dict_path\t: {sample_dict_path}")

    with sample_dict_path.open('rb') as file:
        sample_dict = pickle.load(file)

    
    # print(np.linalg.norm(np.array(sample_dict['inflating'][0][0]) - np.array(sample_dict['inflating'][100][0])))

    print(f'samples\t:{dict_each_length(sample_dict)}')


    head_state = next(iter(sample_dict))
    frame_size = len(sample_dict[head_state][0])
    segment_size = len(sample_dict[head_state][0][0])
    print(f'frame_size\t:{frame_size}')
    print(f'segment_size\t:{segment_size}')


    print('> slice samples.')
    # slice valid range
    # and convert frame into np.array
    print(f'n_sumples\t:{args.n_samples}')
    for name in sample_dict:
        if len(sample_dict[name]) < args.n_samples:
            raise ValueError(f'n_samples ({args.n_samples}) must be heigher than any each samples. '
                             f'name: {name}; sample_length: {len(sample_dict[name])}')
        
        sample_dict[name] = np.array(sample_dict[name][0:args.n_samples])

    # suffle
    for name in sample_dict:
        random.shuffle(sample_dict[name])

    print(f'samples\t: {dict_each_length(sample_dict)}')

    n_states = len(sample_dict)
    shape = [math.ceil(n_states / 2), 2]
    fig, subplots = plt.subplots(*shape)
    for plot_idx, name, frames in zip(itertools.product(*[range(s) for s in shape]), sample_dict.keys(), sample_dict.values()):
        subplot = subplots[plot_idx[0]][plot_idx[1]]
        subplot.set_title(name)
        subplot.plot(frames[0])
    
    fig.tight_layout()
    plt.savefig(context['output_dir'] / 'samples.png')

    print('> divide train and test.')
    # divide train and test 
    train_sample_dict = {name: frames[0:math.floor(args.n_samples/2)] for name, frames in sample_dict.items()}
    test_sample_dict = {name: frames[math.floor(args.n_samples/2) : args.n_samples] for name, frames in sample_dict.items()}

    with (context['output_dir'] / 'train_sample_dict.pkl').open('wb') as file:
        pickle.dump(train_sample_dict, file)

    with (context['output_dir'] / 'test_sample_dict.pkl').open('wb') as file:
        pickle.dump(test_sample_dict, file)

    print(f'train_sample_dict\t: {dict_each_length(train_sample_dict)}')
    print(f'test_sample_dict\t: {dict_each_length(test_sample_dict)}')
    

    # devide train data into each batch
    batches = []
    batch_size = 32

    for idx in range(0, min([ len(frames) for frames in train_sample_dict.values()]) , batch_size):
        batch = {}
        for name, frames in train_sample_dict.items():
            batch[name] = frames[idx: idx + batch_size]
            
        batches.append(batch)

    print(f'n_batches\t: {len(batches)}')
    
    for idx, batch in enumerate(batches):
        print(f'batch_{idx}\t: {dict_each_length(batch)}')

    # train
    model = learning_utils.make_model(frame_size, segment_size)

    print('> evaluate')
    evaluate(model, test_sample_dict)

    for epoch_step in range(args.max_epochs):
        context['epoch_step'] = epoch_step + 1
        print(f"> epoch {context['epoch_step']} / {args.max_epochs}")

        for batch_step, batch in enumerate(batches):
            context['batch_step'] = batch_step + 1
            print(f"> batch {context['batch_step']} / {len(batches)}")

            print('> train')
            train_per_batch(batch, model)

        print('> evaluate')
        evaluate(model, test_sample_dict)

        model.save(context['output_dir'] / f'model-{get_current_step_id(context)}')


    # test

    # print(sample_dict['non'])
    # sample_dict['non'] = sample_dict['non'][0:25]

    # each_length = {name: len(frames) for name, frames in sample_dict.items()}
    # print(f'each_length: \n{each_length}')
    # print(sample_dict['non'])

    # with (sample_dict_path).open('wb') as file:
    #     pickle.dump(sample_dict, file)
    # print(sample_dict)
    print('> end program')

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
    print(f'input, target\t: {len(input_frames)}, {len(target_embeddings)}')
    
    model.fit(input_frames, target_embeddings, batch_size=32)

def evaluate(model, test_sample_dict):
    embedding_dict = {}
    for name, frames in test_sample_dict.items():
        # embedding_dict[name], _ = model(np.array(frames))
        embedding_dict[name] = model(np.array(frames))
    
    for anchor_name, frames in test_sample_dict.items():
        print('> evaluate')
        # at each state, select one index
        anchor_idx = random.randrange(len(frames))

        print(f'anchor\t: {anchor_name}, {anchor_idx}')

        pos_embeddings, pos_emb_locs = get_embeddings(embedding_dict, lambda name, idx: name == anchor_name and idx != anchor_idx )
        neg_embeddings, neg_emb_locs = get_embeddings(embedding_dict, lambda name, idx: name != anchor_name)

        anchor_embedding = embedding_dict[anchor_name][anchor_idx]

        anchor_embedding = np.array(anchor_embedding)
        pos_embeddings = np.array(pos_embeddings)
        neg_embeddings = np.array(neg_embeddings)

        print(f'distance\t: p={np.linalg.norm(anchor_embedding - pos_embeddings)}; n={np.linalg.norm(anchor_embedding - neg_embeddings)}')
        # print(anchor_embedding[0], anchor_embedding[1], anchor_embedding[2])
        # import ipdb; ipdb.set_trace()
        
        print('> plot sample embedding')
        # select one index randomly
        pos_idx = random.randrange(len(pos_embeddings))
        neg_idx = random.randrange(len(neg_embeddings))

        print(f'positive\t: {pos_emb_locs[pos_idx]["name"]}, {pos_emb_locs[pos_idx]["index"]}')
        print(f'negative\t: {neg_emb_locs[neg_idx]["name"]}, {neg_emb_locs[neg_idx]["index"]}')
        pos_emb = pos_embeddings[pos_emb_locs[pos_idx]["index"]]
        neg_emb = neg_embeddings[neg_emb_locs[neg_idx]["index"]]
        fig, subplots = plt.subplots(2, 2)
        subplots[0][0].set_title('all')
        subplots[0][0].plot(neg_emb, label='negative', color='C2')
        subplots[0][0].plot(pos_emb, label='positive', color='C1')
        subplots[0][0].plot(anchor_embedding, label='anchor', color='C0')

        subplots[0][1].set_title('anchor')
        subplots[0][1].plot(anchor_embedding, label='anchor', color='C0')

        subplots[1][0].set_title('positive')
        subplots[1][0].plot(pos_emb, label='positive', color='C1')

        subplots[1][1].set_title('negative')
        subplots[1][1].plot(neg_emb, label='negative', color='C2')
        
        # fig.legend()
        fig.tight_layout()

        plt.savefig(context['output_dir'] / f'evaluate-{get_current_step_id(context)}'
            f'-anchor({anchor_name},{anchor_idx})'
            f'-positive({pos_emb_locs[pos_idx]["name"]},{pos_emb_locs[pos_idx]["index"]})'
            f'-negative({neg_emb_locs[neg_idx]["name"]},{neg_emb_locs[neg_idx]["index"]}).png')

        # print(np.linalg.norm(anchor_embedding, ord=2))
        # print(np.shape(anchor_embedding))
        # break # next state
    

def get_embeddings(embedding_dict, filter_func):
    filtered_embs = []
    locations = []
    for name, embs in embedding_dict.items():
        for idx, emb in enumerate(embs):
            if filter_func(name, idx):
                locations.append({'name': name, 'index': idx})
                filtered_embs.append(emb)
    return filtered_embs, locations


def dict_each_length(d):
    return {key: len(value) for key, value in d.items()}

def get_current_step_id(context):
    return f'epoch-({context["epoch_step"]})-batch-({context["batch_step"]})'

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sample-dict',
        help='Path to the sample_dict.pkl file.',
        default='sample_dict.pkl'
    )

    parser.add_argument(
        '--n-samples',
        help='Number of samples will being used.',
        type=int,
        default=320
    )

    parser.add_argument(
        '--max-epochs',
        help='Number of epochs to train the model.',
        type=int,
        default=10
    )
    # parser.add_argument(
    #     '--batch-size',
    #     help='Number of samples will being used.',
    #     type=int,
    #     default=32
    # )

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
