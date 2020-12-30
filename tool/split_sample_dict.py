import sys
import argparse
import pathlib
import pickle

from stem_lib import stem_utils
from stem_lib.stdlib import file_utils as std_file_utils
from stem_lib.stdlib.logging import TeeLogger

def main(args):

    output_dir = stem_utils.STEM_CONSTANTS.STEM_LOG_DIR / 'split_sample_dict' / std_file_utils.get_unique_log_dir()
    output_dir.mkdir(parents=True)

    sys.stdout = TeeLogger(output_dir / 'stdout.log')

    print(f'output_dir\t: {output_dir}')

    sample_dict_path = pathlib.Path(args.sample_dict)
    print(f"sample_dict_path\t: {sample_dict_path}")
    
    with sample_dict_path.open('rb') as file:
        sample_dict = pickle.load(file)
    
    print(f'samples\t: {dict_each_length(sample_dict)}')

    print(f"names\t: {args.names}")

    names_a = set(args.names)
    names_b = sample_dict.keys() - names_a

    if not names_a <= sample_dict.keys():
        raise ValueError(f"{names_a} is not subset of {sample_dict.keys()} ")
    
    print(f"> split samples into {names_a} and {names_b}")

    part_a = {name: frames for name, frames in sample_dict.items() if name in names_a}
    part_b = {name: frames for name, frames in sample_dict.items() if name in names_b}

    print(f"part_a\t: {dict_each_length(part_a)}")
    print(f"part_b\t: {dict_each_length(part_b)}")

    print(f"> dump")

    with (output_dir / get_sample_dict_filename(part_a)).open('wb') as file:
        pickle.dump(part_a, file)

    with (output_dir / get_sample_dict_filename(part_b)).open('wb') as file:
        pickle.dump(part_b, file)


# def get_set_parts_
def dict_each_length(d):
    return {key: len(value) for key, value in d.items()}

def get_sample_dict_filename(sample_dict):
    return 'sample_dict(' + ','.join(sample_dict.keys()) + ').pkl'

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sample-dict',
        help='Path to the sample_dict.pkl file.',
        required=True
    )

    parser.add_argument(
        '--names',
        nargs='+',
        help='Names for spliting samples into the named samples and the others',
        required=True)

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
