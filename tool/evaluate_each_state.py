import sys
import argparse
import tensorflow as tf
import pickle
import pathlib
import random
import numpy as np
import math
from matplotlib import pyplot as plt
from collections import OrderedDict

from stem_lib import learning_utils
from stem_lib import stem_utils
from stem_lib.stdlib import file_utils as std_file_utils
from stem_lib.stdlib.logging import TeeLogger


def main(args):

    output_dir = stem_utils.STEM_CONSTANTS.STEM_LOG_DIR / pathlib.Path(__file__).stem / std_file_utils.get_unique_log_dir()
    output_dir.mkdir(parents=True)

    sys.stdout = TeeLogger(output_dir / 'stdout.log')
    print(f'output_dir\t: {output_dir}')
    print(f'sys.argv\t: {sys.argv}')

    print('> load model')
    print(f"model_path\t: {args.pretrained_model}")
    model = tf.keras.models.load_model(args.pretrained_model,
                                       custom_objects={'loss': learning_utils.triplet_loss()})

    model.summary()

    print(f'dicts\t: {args.dicts}')

    print(f'> load dicts')
    sample_dict = {}
    for dict_path in args.dicts:
        dict_path = pathlib.Path(dict_path)
        with dict_path.open('rb') as file:
            sample_dict.update(pickle.load(file))
    
    print(f'sample_dict\t: {dict_each_length(sample_dict)}')

    print('> shuffle')
    for name in sample_dict:
        random.shuffle(sample_dict[name])

    print('> slice')
    slice_sample_dict(sample_dict, args.n_samples)


    print(f'sample_dict\t: {dict_each_length(sample_dict)}')
    print(f'frame_sampling_rate\t: {args.frame_sampling_rate}')

    plot_shape = [math.ceil(len(sample_dict) / 2), 2]
    
    for frame_idx in range(args.n_samples):
        print(f'> plot frame {frame_idx} / {args.n_samples}')
        fig = plt.figure()
        
        share_ax = None
        for plot_idx, (name, frames) in enumerate(sample_dict.items()):
            subplot = fig.add_subplot(*plot_shape, plot_idx+1, sharey=share_ax)
            share_ax = subplot
            subplot.set_title(name)
            subplot.set_xlabel('Time (s)')
            frame = frames[frame_idx]
            subplot.plot(np.arange(len(frame))/args.frame_sampling_rate,  frame)
        
        fig.tight_layout()
        plt.savefig(output_dir / f"samples-{frame_idx}.png")
        plt.close()

    print('> embeding')
    emb_dict = {}
    for name, frames in sample_dict.items():
        emb_dict[name] = model(frames)
    
    fig_idx = 0
    for anchor_name, anchor_embs in emb_dict.items():
        print(f'anchor name\t: {anchor_name}')

        fig, ax = plt.subplots()

        means = OrderedDict()
        stds = OrderedDict()

        for name in emb_dict.keys():
            means[name] = []
            stds[name] = []

        for anchor_emb in anchor_embs:

            for comp_name, comp_embs in emb_dict.items():
                distances = np.linalg.norm(anchor_emb - comp_embs, axis=1)

                mean = np.mean(distances)
                std = np.std(distances)
                means[comp_name].append(mean)
                stds[comp_name].append(std)

                print(f'mean\t: {mean}\t; std\t: {std}\t; {comp_name}')
        
        ax.violinplot(means.values())
        fig.savefig(output_dir / f"{fig_idx}-{anchor_name}.png")
        fig_idx = fig_idx+1
        plt.show()

        

    print('> end program')

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
        '--dicts',
        nargs='+',
        required=True
    )

    parser.add_argument(
        '--n-samples',
        help='Number of test samples.',
        type=int,
        default=16
    )

    parser.add_argument(
        '--frame-sampling-rate',
        type=float,
        default=50
    )


    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))