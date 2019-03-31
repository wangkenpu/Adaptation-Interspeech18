#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys


def insert_affine_layer(args):
    nnet3_begin = "<Nnet3> \n"
    with open(args.nnet_in, "r") as fr_nnet_in, \
            open(args.nnet_out, 'w') as fw_nnet_out:
        lines = fr_nnet_in.readlines()
        if nnet3_begin not in lines:
            print("Only support Kaldi Nnet3 format.")
            sys.exit(1)

        pattern_node = re.compile("input=lda")
        pattern_component_num = re.compile("<NumComponents>")
        pattern_nnet3_end = re.compile("</Nnet3>")

        for line in lines:
            match_node = pattern_node.search(line)
            match_component_num = pattern_component_num.search(line)
            match_nnet3_end = pattern_nnet3_end.search(line)
            # Change first hidden layer's input
            if match_node is not None:
                fw_nnet_out.write(
                        "component-node name=lin component=lin input=lda\n")
                new_line = pattern_node.sub("input=lin", line)
                fw_nnet_out.write(new_line)
                continue
            # Change component num
            elif match_component_num is not None:
                line = line.strip().split()
                assert len(line) == 2
                line[1] = str(int(line[1]) + 1)
                line = line[0] + " " + line[1] + " \n"
                fw_nnet_out.write(line)
            elif match_nnet3_end is not None:
                make_affine_component(args.lda_dim, fw_nnet_out)
                fw_nnet_out.write(line)
            else:
                fw_nnet_out.write(line)



def make_affine_component(in_dim, nnet_out):
    # Insert LIN layer
    component_name = "<ComponentName> lin "
    component_type = "<NaturalGradientAffineComponent> "
    max_change = "<MaxChange> 0.75 "
    learning_rate = "<LearningRate> 0.001 "
    linear_params = "<LinearParams>  "
    bias_params = "<BiasParams>  "
    rank_in = "<RankIn> 20 "
    rank_out = "<RankOut> 80 "
    update_period = "<UpdatePeriod> 4 "
    num_samples_history = "<NumSamplesHistory> 2000 "
    alpha = "<Alpha> 4 "
    is_gradient = "<IsGradient> F "
    component_type_end = "</NaturalGradientAffineComponent> \n"
    nnet_out.write(component_name)
    nnet_out.write(component_type)
    nnet_out.write(max_change)
    nnet_out.write(learning_rate)
    nnet_out.write(linear_params)
    make_linear_params(in_dim, nnet_out)
    nnet_out.write(bias_params)
    make_bias_params(in_dim, nnet_out)
    nnet_out.write(rank_in)
    nnet_out.write(rank_out)
    nnet_out.write(update_period)
    nnet_out.write(num_samples_history)
    nnet_out.write(alpha)
    nnet_out.write(is_gradient)
    nnet_out.write(component_type_end)


def make_linear_params(in_dim, nnet_out):
    # Initialize LIN layer as an identity matrix
    nnet_out.write("[\n")
    for row in range(in_dim):
        nnet_out.write("  ")
        for col in range(in_dim):
            if row == col:
                nnet_out.write("1.0 ")
            else:
                nnet_out.write("0.0 ")
        if row == in_dim - 1:
            nnet_out.write("]\n")
        else:
            nnet_out.write("\n")


def make_bias_params(in_dim, nnet_out):
    # Initialize LIN bias vector
    nnet_out.write("[ ")
    for col in range(in_dim):
        nnet_out.write("0.0 ")
    nnet_out.write("]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="%(prog)s",
        description="Insert an affine-layer after LDA transform."
    )
    parser.add_argument(
        "--lda_dim",
        type=int,
        default=300,
        help="The feature dim after LDA transform."
    )
    parser.add_argument(
        "--nnet_in",
        type=str,
        required=True,
        help="Input nnet which needed to be insert an affine layer."
    )
    parser.add_argument(
        "--nnet_out",
        type=str,
        default="0.mdl",
        help="Output network after inserting affine layer."
    )

    FLAGS, unparsed = parser.parse_known_args()
    insert_affine_layer(FLAGS)
