#!/usr/bin/env python3
import tensorflow as tf                 # tf.__version__ : 1.4
import numpy as np
import scipy.misc
import argparse
import os, sys

import models
from models import *

# ---[ Session Configures
GpuConfig = tf.ConfigProto()
GpuConfig.gpu_options.allow_growth=True
#GpuConfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


MODEL_NAME = 'cyclegan'

# ---[ Args
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Photos convert to segmentation lables via CycleGAN")
    parser.add_argument("model", type=str, choices=models.__list, 
                                            help="model choices. [%(default)s]")

    parser.add_argument("--source_dataset", 
                                            type=str,  default=None,
                                            help="<data_dir>:<list_filename> [%(default)s]")

    parser.add_argument("--bs",             type=int,  default=12,
                                            help="[%(default)s]")
    parser.add_argument("--crop_size",      type=str,  default='713,713',
                                            help="[%(default)s]")
    parser.add_argument("--resize",         type=str,  default='256,256',
                                            help="[%(default)s]")
    parser.add_argument("--print_epoch",    type=int,  default=100,
                                            help="[%(default)s]")
    parser.add_argument("--max_epoch",      type=int,  default=200000,
                                            help="[%(default)s]")
    parser.add_argument("--g_lr",           type=float,  default=0.0002,
                                            help="[%(default)s]")
    parser.add_argument("--d_lr",           type=float,  default=0.0002,
                                            help="[%(default)s]")
    parser.add_argument("--g_epoch",        type=int,  default=1,
                                            help="[%(default)s]")
    parser.add_argument("--d_epoch",        type=int,  default=1,
                                            help="[%(default)s]")
    parser.add_argument("--optimizer",      type=dict,  default=dict(eval='AdamOptimizer', args={'beta1':0., 'beta2':0.9}),
                                            help="(coming soon...) [%(default)s]")
    parser.add_argument("--lm",             type=float,  default=10.0,
                                            help="L1 lambda for D G loss [%(default)s]")
    parser.add_argument("--gpus",           type=int,  default=4,
                                            help="[%(default)s]")
    #parser.add_argument("--num_threads",    type=int,  default=32,
                                            #help="[%(default)s]")
    parser.add_argument("--result_dir",     type=str,  default='results/',
                                            help="[%(default)s]")
    parser.add_argument("--summary_dir",    type=str,  default='summary/',
                                            help="[%(default)s]")
    parser.add_argument("--name",           type=str,  default=None,
                                            help="[%(default)s]")
    return parser.parse_args()

def main():
    args = get_arguments()
    config = args.__dict__
    config['bs'] = max(config['bs'], 1)
    keys = [k for k in config.keys() if not config[k]]
    for k in keys:
        del config[k]

    print ("Configures:")
    for k,v in config.items():
        print ("\t-- %-24s : %-s" % (k, v))

    # ---[ Loading data
    MODEL = config['model']
    model = eval(MODEL)(**config)
    model.build()


if __name__ == '__main__':
    main()

