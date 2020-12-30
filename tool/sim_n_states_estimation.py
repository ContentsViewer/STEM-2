import sys
import argparse
import pickle
import random
import pathlib
import itertools
import math
import numpy as np
from collections import deque
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

    context['output_dir'] = stem_utils.STEM_CONSTANTS.STEM_LOG_DIR / 'sim_n_states_estimation' / std_file_utils.get_unique_log_dir()
    context['output_dir'].mkdir(parents=True)

    sys.stdout = TeeLogger(context['output_dir'] / 'stdout.log')

    print(f'output_dir\t:{context["output_dir"]}')


    sample_dict_path = pathlib.Path(args.sample_dict)

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

    print(f'> shuffle')
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


    model = learning_utils.make_model(frame_size, segment_size)

    sequence = [{'name': 'non'      , 'frames': sample_dict['non'][0:10]},
                {'name': 'inflating', 'frames': sample_dict['inflating'][0:10]},
                {'name': 'shrinking', 'frames': sample_dict['shrinking'][0:10]},
                {'name': 'baunded', 'frames': sample_dict['baunded'][0:10]}]

    
    state_embedding_memory = []            
    
    sensitivity = 0.4
    
    # first memory
    first_frame = np.array([sample_dict['non'][0]])
    prev_emb = model(first_frame)

    state_embedding_memory.append({'frames': deque([first_frame]), 'embeddings': deque([prev_emb])})

    current_state_idx = 0
    is_prev_state_unknown = False

    for span in sequence:
        print(f"span\t: {span['name']}")
        for frame in span['frames']:
            curr_emb = model(np.array([frame]))
            diff_norm = np.linalg.norm(curr_emb - prev_emb)
            print(f"diff_norm\t: {diff_norm}")

            if diff_norm > sensitivity:
                print(f"> detect state change")

                emb_distances = get_embedding_distance(curr_emb, state_embedding_memory)
                print(f"distances\t:{emb_distances}")

                print(f"is_prev_state_unknown\t: {is_prev_state_unknown}")

                if is_prev_state_unknown:
                    print(f"> train prev state")
                    is_prev_state_unknown = False

                else:
                    state_embedding_memory.append({'frames': deque([frame]), 'embeddings': deque([curr_emb])})
                    current_state_idx += 1
                    is_prev_state_unknown = True
            else:
                print(f"> same state")
                state_embedding_memory[current_state_idx]['frames'].append(frame)
                state_embedding_memory[current_state_idx]['embeddings'].append(curr_emb)

            prev_emb = curr_emb
    
    # embs_non = model(sample_dict['non'])

    # embs_inflating = model(sample_dict['inflating'])

    # embs_shrinking = model(sample_dict['shrinking'])

    # embs_bounded = model(sample_dict['bounded'])


def re_train(model, state_embedding_memory):
    print('> re-train')

    input_frames = []
    target_embs = []
    for anchor_state_idx, memory in enumerate(state_embedding_memory):
        for anchor_idx, anchor_emb in enumerate(memory['embeddings']):
            pos_embs, pos_locs = get_embeddings(state_embedding_memory,
                                                lambda state_idx, memory_idx: state_idx == anchor_state_idx and memory_idx != anchor_idx)
            
            neg_embs, neg_locs = get_embeddings(state_embedding_memory,
                                                lambda state_idx, memory_idx: state_idx != anchor_state_idx)
            

            triplets = learning_utils.select_triplets(anchor_emb, pos_embs, neg_embs)

            for anchor_emb, pos_emb, neg_emb in triplets:
                input_frames.append(memory['frames'][anchor_idx])
                target_embs.append([pos_emb, neg_emb])
    
    input_frames = np.array(input_frames)
    target_embs = np.array(target_embs)

    print(f'input, target\t: {len(input_frames)}, {len(target_embs)}')
    
    model.fit(input_frames, target_embs, batch_size=32)

def get_embedding_distance(embedding, state_embedding_memory):
    distances = []
    for state_idx, memory in enumerate(state_embedding_memory):
        distance = np.linalg.norm(embedding - np.array(memory['embeddings']))
        distances.append(distance)
    return distances

def get_embeddings(state_embedding_memory, filter_func):
    filtered_embs = []
    locations = []
    for state_idx, memory in enumerate(state_embedding_memory):
        for memory_idx, emb in enumerate(memory['embeddings']):
            if filter_func(state_idx, memory_idx):
                locations.append({'state_idx': state_idx, 'memory_idx': memory_idx})
                filtered_embs.append(emb)
    
    return filtered_embs, locations

def dict_each_length(d):
    return {key: len(value) for key, value in d.items()}


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
        default=200
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
