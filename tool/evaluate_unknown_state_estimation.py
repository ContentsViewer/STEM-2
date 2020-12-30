import sys
import argparse
import tensorflow as tf
import pickle
import pathlib
import random
import numpy as np
import itertools
import math
from matplotlib import pyplot as plt

from stem_lib import learning_utils
from stem_lib import stem_utils
from stem_lib.stdlib import file_utils as std_file_utils
from stem_lib.stdlib.logging import TeeLogger


def main(args):

    output_dir = stem_utils.STEM_CONSTANTS.STEM_LOG_DIR / pathlib.Path(__file__).stem / std_file_utils.get_unique_log_dir()
    output_dir.mkdir(parents=True)

    sys.stdout = TeeLogger(output_dir / 'stdout.log')
    print(f'output_dir\t: {output_dir}')

    print('> load model')
    print(f"model_path\t: {args.pretrained_model}")
    model = tf.keras.models.load_model(args.pretrained_model,
                                       custom_objects={'loss': learning_utils.triplet_loss()})

    model.summary()

    known_sample_dict_path = pathlib.Path(args.known_sample_dict)
    print('> load known_sample_dict')
    print(f"known_sample_dict_path\t: {known_sample_dict_path}")
    with known_sample_dict_path.open('rb') as file:
        known_sample_dict = pickle.load(file)
    print(f"known_sample_dict\t: {dict_each_length(known_sample_dict)}")
    
    
    unknown_sample_dict_path = pathlib.Path(args.unknown_sample_dict)
    print('> load unknown_sample_dict')
    print(f"unknown_sample_dict_path\t: {unknown_sample_dict_path}")
    with unknown_sample_dict_path.open('rb') as file:
        unknown_sample_dict = pickle.load(file)
    print(f"unknown_sample_dict\t: {dict_each_length(unknown_sample_dict)}")

    print('> shuffle')
    for name in known_sample_dict:
        random.shuffle(known_sample_dict[name])
    for name in unknown_sample_dict:
        random.shuffle(unknown_sample_dict[name])
    
    print('> slice')
    slice_sample_dict(known_sample_dict, args.n_samples)
    slice_sample_dict(unknown_sample_dict, args.n_samples)


    print(f"known_sample_dict\t: {dict_each_length(known_sample_dict)}")
    print(f"unknown_sample_dict\t: {dict_each_length(unknown_sample_dict)}")


    n_states = len(known_sample_dict) + len(unknown_sample_dict)
    shape = [math.ceil(n_states / 2), 2]

    for frame_idx in range(args.n_samples):
        fig, subplots = plt.subplots(*shape)
        plot_idx_iter = itertools.product(*[range(s) for s in shape])

        for name, frames in itertools.chain(known_sample_dict.items(), unknown_sample_dict.items()):
            plot_idx = next(plot_idx_iter)
            subplot = subplots[plot_idx[0]][plot_idx[1]]
            subplot.set_title(name)
            subplot.plot(frames[frame_idx])
        
        fig.tight_layout()
        plt.savefig(output_dir / f"samples-{frame_idx}.png")

    known_emb_dict = {}
    for name, frames in known_sample_dict.items():
        known_emb_dict[name] = model(frames)

    unknown_emb_dict = {}
    for name, frames in unknown_sample_dict.items():
        unknown_emb_dict[name] = model(frames)
    

    for anchor_name, frames in unknown_sample_dict.items():
        print(f"anchor_name\t: {anchor_name}")
        for frame_idx, frame in enumerate(frames):
            # each frame
            print(f"frame_idx\t: {frame_idx}")
            
            anchor_emb = unknown_emb_dict[anchor_name][frame_idx]

            fig, subplots = plt.subplots(*shape)
            plot_idx_iter = itertools.product(*[range(s) for s in shape])
            
            compare_embeddings(anchor_emb, unknown_emb_dict, subplots, plot_idx_iter)
            compare_embeddings(anchor_emb, known_emb_dict, subplots, plot_idx_iter)

            fig.tight_layout()
            plt.savefig(output_dir / f"distances-{anchor_name}-{frame_idx}.png")


            
def compare_embeddings(anchor_emb, comp_emb_dict, subplots, plot_idx_iter):

    for comp_name, comp_embs in comp_emb_dict.items():
        print(f"comp_name\t: {comp_name}")
        
        distances = np.linalg.norm(anchor_emb - comp_embs, axis=1)
        print(f"distances\t: {distances}")

        mean_distance = np.mean(distances)
        print(f"mean_distance\t: {mean_distance}")

        plot_idx = next(plot_idx_iter)
        subplot = subplots[plot_idx[0]][plot_idx[1]]
        subplot.set_title(comp_name)
        subplot.set_ylim([0, 0.1])
        subplot.axhline(mean_distance, linestyle='--', color='C1')
        subplot.plot(distances, color='C0')


def slice_sample_dict(sample_dict, slice_size):
    for name in sample_dict:
        if len(sample_dict[name]) < slice_size:
            raise ValueError(f'slice_size ({slice_size}) must be less than any each frame length. '
                             f'name: {name}; frame_length: {len(sample_dict[name])}')
        
        sample_dict[name] = np.array(sample_dict[name][0:slice_size])


def dict_each_length(d):
    return {key: len(value) for key, value in d.items()}


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pretrained-model',
        help='Load a pretrained model.',
        required=True
    )

    parser.add_argument(
        '--known-sample-dict',
        help='Path to the sample_dict.pkl file which contains states KNOWN by the pretraied model.',
        required=True
    )
    
    parser.add_argument(
        '--unknown-sample-dict',
        help='Path to the sample_dict.pkl file which contains states UNKNOWN by the pretraied model.',
        required=True
    )
    
    parser.add_argument(
        '--n-samples',
        help='Number of test samples.',
        type=int,
        default=16
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
