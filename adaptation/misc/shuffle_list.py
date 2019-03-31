#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Shuffle kaldi scp file."""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import pprint
import random
import sys


def main():
    """Shuffle file and save"""
    with open(FLAGS.inputs_scp, 'r') as fr_inputs, \
            open(FLAGS.outputs_scp, 'w') as fw_outputs:
        lists_inputs = fr_inputs.readlines()
        lists = range(len(lists_inputs))
        random.seed(0)
        random.shuffle(lists)
        for i in range(len(lists)):
            line_input = lists_inputs[lists[i]]
            fw_outputs.write(line_input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage="%(prog)s input_file output_file",
        description="Randomizes the order of lines of input."
    )
    parser.add_argument(
        "inputs_scp",
        type=str,
        help="input filename"
    )
    parser.add_argument(
        "outputs_scp",
        type=str,
        help="output filename"
    )
    FLAGS, unparsed = parser.parse_known_args()
    # pp = pprint.PrettyPrinter()
    # pp.pprint(FLAGS.__dict__)
    main()
