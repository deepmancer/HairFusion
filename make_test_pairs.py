import os

import argparse

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_path', type=str, required=True,
                        help='path of image dir to make test pairs. ex) ./data/Custom')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dir_path = args.dir_path

    root = os.path.join(dir_path, 'images')

    img_list = os.listdir(root)
    f = open(os.path.join(dir_path, 'test_pairs.txt'), 'w')


    for img_name1 in img_list:
        for img_name2 in img_list:
            data = f"{img_name1} {img_name2}\n"
            f.write(data)
    f.close()
