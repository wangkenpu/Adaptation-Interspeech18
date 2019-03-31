#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys

import numpy as np


def insert_lhuc_layer(args):
    hmm_begin = "<TransitionModel> \n"
    nnet3_begin = "<Nnet3> \n"
    with open(args.nnet_in, "r") as fr_nnet_in:
        lines = fr_nnet_in.readlines()
        if (nnet3_begin not in lines) or (hmm_begin not in lines):
            print("Only support Kaldi Nnet3 text model (*.mdl), exit now.")
            sys.exit(1)

    dim_recorder = get_component_dim(args.nnet_in)
    tmp_insert_file_name = "/tmp/insert_lhuc.txt"
    with open(tmp_insert_file_name, "w"): pass

    pattern_tdnn_end = re.compile("component-node name=tdnn(\d)*\.renorm")
    pattern_lstm_end = re.compile("dim-range-node name=lstm(\d)*\.r_trunc")
    pattern_component_num = re.compile("<NumComponents>")
    pattern_nnet3_end = re.compile("</Nnet3>")

    length = len(lines)
    index = 0
    num_lhuc = 0

    with open(args.nnet_out, 'w') as fw_nnet_out:
        while True:
            if index >= length:
                break
            line = lines[index]
            match_tdnn = pattern_tdnn_end.search(line)
            match_lstm = pattern_lstm_end.search(line)
            match_component_num = pattern_component_num.search(line)
            match_nnet3_end = pattern_nnet3_end.search(line)
            if (match_tdnn is not None) or (match_lstm is not None):
                if match_tdnn is not None:
                    layer_name = match_tdnn.group()
                elif match_lstm is not None:
                    layer_name = match_lstm.group()
                # print("{} {}".format(index+1, line), end="")
                fw_nnet_out.write(line)
                layer_name = layer_name.strip().split()[-1]
                layer_name = layer_name.split("=")[-1]
                layer_name = layer_name.split(".")[0]
                num_lhuc = num_lhuc + 1
                insert_lhuc_component_node(layer_name, num_lhuc, dim_recorder, 
                                           tmp_insert_file_name, fw_nnet_out)
                index = index + 1
                line = lines[index]
                # print(line)
                if match_tdnn is not None:
                    pattern_sub = re.compile("tdnn(\d)*\.renorm")
                elif match_lstm is not None:
                    pattern_sub = re.compile("lstm(\d)*\.rp")
                else:
                    print("Unknow pattern.")
                    sys.exit(1)
                new_string = "LHUC{}.dot".format(num_lhuc)
                result = pattern_sub.sub(new_string, line)
                fw_nnet_out.write(result)
            elif match_component_num is not None:
                line = line.strip().split()
                assert len(line) == 2
                line[1] = str(int(line[1]) + num_lhuc * 4)
                line = line[0] + " " + line[1] + " \n"
                fw_nnet_out.write(line)
            elif match_nnet3_end is not None:
                with open(tmp_insert_file_name, "r") as fr_insert:
                    insert_lines = fr_insert.readlines()
                for insert_line in insert_lines:
                    fw_nnet_out.write(insert_line)
                fw_nnet_out.write(line)
            else:
                fw_nnet_out.write(line)
            index = index + 1
        os.remove(tmp_insert_file_name)


def get_component_dim(model):
    with open(model) as fr_model:
        lines = fr_model.readlines()

    pattern_tdnn = re.compile("<ComponentName> tdnn(\d)*\.affine")
    pattern_lstm = re.compile("<ComponentName> lstm(\d)*\.W_rp")
    pattern_bias = re.compile("<BiasParams>")

    dim_recorder = {}
    length = len(lines)
    index = 0

    while True:
        if index >= length:
            break
        line = lines[index]
        match_tdnn = pattern_tdnn.search(line)
        match_lstm = pattern_lstm.search(line)
        if (match_tdnn is not None) or (match_lstm is not None):
            if match_tdnn is not None:
                name = match_tdnn.group().strip().split()[-1]
            elif match_lstm:
                name = match_lstm.group().strip().split()[-1]
            else:
                print("Unknow match.")
                sys.exit(1)
            index = index + 1
            while True:
                line = lines[index]
                match_bias = pattern_bias.search(line)
                if match_bias is not None:
                    dim = len(line.strip().split()) - 3
                    assert not dim_recorder.has_key(name)
                    dim_recorder[name] = dim
                    break
                else:
                    index = index + 1
        else:
            pass
        index = index + 1
    return dim_recorder


def insert_lhuc_component_node(layer_name, lhuc_index, dim_recorder, 
                               tmp_insert_file, fw_nnet_out):
    if "tdnn" in layer_name:
        layer_input = layer_name + ".renorm"
        layer_name_lhuc = layer_name + ".affine"
    elif "lstm" in layer_name:
        layer_input = layer_name + ".rp"
        layer_name_lhuc = layer_name + ".W_rp"
    lhuc_name = "LHUC{}".format(lhuc_index)
    dim = dim_recorder[layer_name_lhuc]

    fw_nnet_out.write("component-node name={name}.r component={name}.r "
        "input={input}\n".format(name=lhuc_name, input=layer_input))
    fw_nnet_out.write("component-node name={name}.sigmoid component={name}.sigmoid "
        "input={name}.r\n".format(name=lhuc_name))
    fw_nnet_out.write("component-node name={name}.scale component={name}.scale "
        "input={name}.sigmoid\n".format(name=lhuc_name))
    fw_nnet_out.write("component-node name={name}.dot component={name}.dot "
        "input=Append({name}.scale, {input})\n".format(
              name=lhuc_name, input=layer_input))
    make_constant_component(lhuc_name, dim, tmp_insert_file)


def make_constant_component(name, dim, tmp_insert_file):
    with open(tmp_insert_file, "a") as fw:
        # ConstantComponent
        fw.write("<ComponentName> {}.r <ConstantComponent> "
                 "<LearningRate> 0.001 <Output>  ".format(name))
        matrix = make_lhuc_parameters(dim)
        assert matrix.size == dim
        fw.write("[ ")
        for i in range(dim):
            fw.write("{0:e} ".format(matrix[i]))
        fw.write("]\n")
        fw.write("<IsUpdatable> T <UseNaturalGradient> T </ConstantComponent> \n")
        # SigmoidComponent
        fw.write("<ComponentName> {}.sigmoid <SigmoidComponent> <Dim> {} "
                 "<ValueAvg>  [ ]\n".format(name, dim))
        fw.write("<DerivAvg>  [ ]\n")
        fw.write("<Count> 0 <NumDimsSelfRepaired> 0 <NumDimsProcessed> 0 "
                 "</SigmoidComponent> \n")
        # PerElementScaleComponent
        fw.write("<ComponentName> {}.scale <PerElementScaleComponent> "
                 "<LearningRateFactor> 0 <LearningRate> 0 "
                 "<Params>  ".format(name))
        fw.write("[ ")
        for i in range(dim):
            fw.write("2 ")
        fw.write("]\n")
        fw.write("</PerElementScaleComponent> \n")
        # ElementwiseProductComponent
        fw.write("<ComponentName> {}.dot <ElementwiseProductComponent> "
                 "<InputDim> {} <OutputDim> {} "
                 "</ElementwiseProductComponent> \n".format(name, dim*2, dim))


def make_lhuc_parameters(dim):
    mu = 0.0
    sigma = 0.001
    np.random.seed(0)
    s = np.random.normal(mu, sigma, dim)
    return s




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="%(prog)s",
        description="Insert LHUC layer after every hidden layer."
    )
    parser.add_argument(
        "--nnet-in",
        type=str,
        required=True,
        help="Input model which needed to be insert LHUC layers."
    )
    parser.add_argument(
        "--nnet-out",
        type=str,
        default="0.mdl",
        help="output network model after inserting LHUC layers."
    )

    FLAGS, unparsed = parser.parse_known_args()

    insert_lhuc_layer(FLAGS)
