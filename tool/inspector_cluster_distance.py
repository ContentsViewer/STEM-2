import sys
import argparse
import tensorflow as tf
import pickle
import pathlib
import numpy as np
import math
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
    

    distances_map = OrderedDict()
    for anchor_name, anchor_embs in emb_dict.items():
        print(f'anchor name\t: {anchor_name}')

        distances_map[anchor_name] = OrderedDict()

        for name in emb_dict.keys():
            distances_map[anchor_name][name] = []
        
        for anchor_emb in anchor_embs:
            for comp_name, comp_embs in emb_dict.items():
                distances_map[anchor_name][comp_name].extend(np.linalg.norm(anchor_emb - comp_embs, axis=1))
        

    print(f'distances_map: ')
    for anchor_name, comp_distances in distances_map.items():
        print(f'  anchor_name: {anchor_name}')
        for comp_name, distances in comp_distances.items():
            print(f'    comp_name: {comp_name}')
            print(f'      len(distances): {len(distances)}')
            comp_distances[comp_name] = np.array(distances)

    print('> READY')
    from IPython.core.debugger import Pdb; Pdb().set_trace()
    print('> end program')


def evaluate_percentile(q, distances_map, anchor_name):
    print(f'> evaluate_percentile')
    print(f'anchor_name\t: {anchor_name}')

    thres=np.percentile(distances_map[anchor_name][anchor_name],q)
    print(f'thres\t: {thres}')

    for comp_name, distances in distances_map[anchor_name].items():
        print(f'comp_name\t: {comp_name}')
        count=np.count_nonzero(distances<thres)
        print(f'count(distances<{thres})\t: {count}')
        print(f'{count}/{len(distances)}\t: {count / len(distances)}')



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