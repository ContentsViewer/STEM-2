import sys
import argparse
import pickle
import pathlib


def main(args):

    sample_dict_path = pathlib.Path(args.sample_dict)

    with (sample_dict_path).open('rb') as file:
        sample_dict = pickle.load(file)
    
    print(sample_dict)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sample-dict',
        help="Path to the sample_dict.pkl file.",
        default='.stem/samples/sample_dict.pkl'
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
