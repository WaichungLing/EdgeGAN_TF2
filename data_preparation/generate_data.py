import glob
import os
from argparse import ArgumentParser
import numpy as np
import scipy.misc

from edgegan.utils import makedirs

# Create the dataset as the format of https://github.com/sysu-imsl/EdgeGAN/tree/master/images/dataset_example
# This script will create a multiclass dataset from sketchyCOCO with 5 classes (sheep, cat, giraffe, zebra, dog)

# In SketchyCOCO dataset, the indices for (sheep, cat, giraffe, zebra, dog) are (20, 17, 25, 24, 18)

# output_dir = '../images/data'
output_dir = './oldimages/data'

def transform(img, out_dim=64):
    bg = np.ones((out_dim, out_dim, 3)) * 255
    max_d = max(img.shape[0], img.shape[1])
    if max_d >= out_dim:
        ratio = img.shape[0] / out_dim
        if max_d == img.shape[0]:
            out_dim_2 = int(img.shape[1] / ratio)
            ret = scipy.misc.imresize(img, (out_dim, out_dim_2))
            bg[:, (out_dim - out_dim_2) // 2:(out_dim - out_dim_2) // 2 + out_dim_2, :] = ret
        else:
            out_dim_2 = int(img.shape[0] / ratio)
            ret = scipy.misc.imresize(img, (out_dim_2, out_dim))
            bg[(out_dim - out_dim_2) // 2:(out_dim - out_dim_2) // 2 + out_dim_2, :, :] = ret
    else:
        bg[(out_dim - img.shape[0]) // 2:(out_dim - img.shape[0]) // 2 + img.shape[0],
        (out_dim - img.shape[1]) // 2:(out_dim - img.shape[1]) // 2 + img.shape[1], :] = img
    return bg


def generate_train(classes, args):
    # Generate our dataset
    # gt_dir = './Animals/GT'
    # edge_dir = './Animals/Edge'

    # Generate old dataset
    gt_dir = './Data/GT'
    edge_dir = './Data/Edge'
    for c in classes:
        c_dir = os.path.join(gt_dir, 'train/'+ str(c) + '/*.png')
        print(c_dir)

        out_c_dir = os.path.join(output_dir, 'train/' + str(c))
        makedirs(out_c_dir)  # Create output directory

        for filename in glob.glob(c_dir):
            f = filename[filename.rfind('/') + 1:]

            print(f)

            img = scipy.misc.imread(filename, mode='RGB')
            trans_img = transform(img, args.out_dim)

            edge = scipy.misc.imread(os.path.join(edge_dir, 'train/'+str(c) + '/' + f), mode='RGB')
            trans_edge = transform(edge, args.out_dim)

            trans = np.concatenate((trans_edge, trans_img), axis=1)
            scipy.misc.imsave(os.path.join(out_c_dir, f), trans)


def generate_test(classes, args):
    sketch_dir = '../Data/Sketch/val'
    idx = 0
    for c in classes:
        c_dir = os.path.join(sketch_dir, str(c) + '/*.png')
        out_c_dir = os.path.join(output_dir, 'test/' + str(idx))
        makedirs(out_c_dir)
        idx += 1

        for filename in glob.glob(c_dir):
            f = filename[filename.rfind('/') + 1:]

            sk = scipy.misc.imread(filename, mode='RGB')
            trans_sk = transform(sk, args.out_dim)

            trans = np.concatenate((trans_sk, trans_sk), axis=1)
            scipy.misc.imsave(os.path.join(out_c_dir, f), trans)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_dim', type=int, default=64, help='the dimension of the output image')
    args = parser.parse_args()
    generate_train([17, 18, 20, 24, 25], args)
    # generate_test([17, 18, 20, 24, 25], args)

