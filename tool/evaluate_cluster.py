import sys
import argparse
import tensorflow as tf
import pickle
import pathlib
import random
import numpy as np
import math
import itertools
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

    print('> embeding')
    emb_dict = {}
    for name, frames in sample_dict.items():
        emb_dict[name] = model(frames)
    

    fig, ax = plt.subplots()

    distances_map = OrderedDict()

    for (anchor_name, comp_name) in itertools.combinations_with_replacement(emb_dict.keys(), 2):
        print(f'anchor\t: {anchor_name}; comp\t: {comp_name}')
        distances = []
        for anchor_emb in emb_dict[anchor_name]:
            distances.extend(np.linalg.norm(anchor_emb - emb_dict[comp_name], axis=1))
        
        print(f'len(distances)\t: {len(distances)}')
        distances_map[(anchor_name, comp_name)] = distances

    ax.violinplot(distances_map.values(), showmeans=True)
    ax.set_xticks(list(range(1, len(distances_map) + 1)))
    ax.set_xticklabels([f'{anchor}\n-{comp}' for (anchor, comp) in distances_map.keys()])
    ax.set_ylabel('Distance')
    fig.savefig(output_dir / f"result-total.png")
    plt.show()


    ####

    plot_shape = [math.ceil(len(sample_dict) / 2), 2]
    xticks = list(range(1, len(emb_dict) + 1))
    fig = plt.figure()
    share_ax = None

    for plot_idx, (anchor_name, anchor_embs) in enumerate(emb_dict.items()):
        print(f'anchor name\t: {anchor_name}')


        distances_map = OrderedDict()

        for name in emb_dict.keys():
            distances_map[name] = []

        for anchor_emb in anchor_embs:

            for comp_name, comp_embs in emb_dict.items():
                distances_map[comp_name].extend(np.linalg.norm(anchor_emb - comp_embs, axis=1))
        
        for distances in distances_map.values():
            print(f'len(distances)\t: {len(distances)}')

        subplot = fig.add_subplot(*plot_shape, plot_idx + 1, sharey=share_ax)
        share_ax = subplot
        subplot.violinplot(distances_map.values(), showmeans=True)
        subplot.set_title(anchor_name)
        subplot.set_xticks(xticks)
        subplot.set_xticklabels(distances_map.keys())
        subplot.set_ylabel('Distance')
    
    fig.tight_layout()
    fig.savefig(output_dir / f"result-each.png")
    plt.show()

    print('> end program')

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

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))