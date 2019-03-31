// nnet3/nnet-training.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)
//              2016  Xiaohui Zhang
//              2017  Ke Wang       Xiaomi Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET3_NNET_KLD_TRAINING_H_
#define KALDI_NNET3_NNET_KLD_TRAINING_H__

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-example-utils.h"
#include "nnet3/nnet-training.h"

namespace kaldi {
namespace nnet3 {

/** This class is for single-threaded training of neural nets using
    standard objective functions such as cross-entropy (implemented with
    logsoftmax nonlinearity and a linear objective function) and quadratic loss.

    Something that we should do in the future is to make it possible to have
    two different threads, one for the compilation, and one for the computation.
    This would only improve efficiency in the cases where the structure of the
    input example was different each time, which isn't what we expect to see in
    speech-recognition training.  (If the structure is the same each time,
    the CachingOptimizingCompiler notices this and uses the computation from
    last time).
 */
class NnetKLDTrainer {
 public:
  NnetKLDTrainer(const NnetTrainerOptions &config,
                 const float32 rho, Nnet *si_nnet, Nnet *nnet);

  // train on one minibatch.
  void Train(const NnetExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // Prints out the max-change stats (if nonzero): the percentage of time that
  // per-component max-change and global max-change were enforced.
  void PrintMaxChangeStats() const;

  ~NnetKLDTrainer();
 private:
  // The internal function for doing one step of conventional SGD training.
  void TrainInternal(const NnetExample &eg,
                     const NnetComputation &si_computation,
                     const NnetComputation &computation);

  void ProcessOutputs(const NnetExample &eg,
                      NnetComputer *si_computer,
                      NnetComputer *computer);

  const NnetTrainerOptions config_;
  Nnet *nnet_;
  Nnet *delta_nnet_;  // nnet representing parameter-change for this minibatch
                      // (or, when using momentum, the moving weighted average
                      // of this).
  const float32 rho_;       // KLD weight
  const Nnet *si_nnet_;     // speaker independent nnet model.
  CachingOptimizingCompiler compiler_;
  CachingOptimizingCompiler si_compiler_;

  // This code supports multiple output layers, even though in the
  // normal case there will be just one output layer named "output".
  // So we store the objective functions per output layer.
  int32 num_minibatches_processed_;

  // stats for max-change.
  std::vector<int32> num_max_change_per_component_applied_;
  int32 num_max_change_global_applied_;

  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher> objf_info_;

  // This value is used in backstitch training when we need to ensure
  // consistent dropout masks.  It's set to a value derived from rand()
  // when the class is initialized.
  int32 srand_seed_;
};

/**
   This function computes the objective function, and if supply_deriv = true,
   supplies its derivative to the NnetComputation object.
   See also the function ComputeAccuracy(), declared in nnet-diagnostics.h.

  @param [in]  supervision   A GeneralMatrix, typically derived from a NnetExample,
                             containing the supervision posteriors or features.
  @param [in] objective_type The objective function type: kLinear = output *
                             supervision, or kQuadratic = -0.5 * (output -
                             supervision)^2.  kLinear is used for softmax
                             objectives; the network contains a LogSoftmax
                             layer which correctly normalizes its output.
  @param [in] output_name    The name of the output node (e.g. "output"), used to
                             look up the output in the NnetComputer object.

  @param [in] supply_deriv   If this is true, this function will compute the
                             derivative of the objective function and supply it
                             to the network using the function
                             NnetComputer::AcceptOutputDeriv
  @param [in,out] computer   The NnetComputer object, from which we get the
                             output using GetOutput and to which we may supply
                             the derivatives using AcceptOutputDeriv.
  @param [out] tot_weight    The total weight of the training examples.  In the
                             kLinear case, this is the sum of the supervision
                             matrix; in the kQuadratic case, it is the number of
                             rows of the supervision matrix.  In order to make
                             it possible to weight samples with quadratic
                             objective functions, we may at some point make it
                             possible for the supervision matrix to have an
                             extra column containing weights.  At the moment,
                             this is not supported.
  @param [out] tot_objf      The total objective function; divide this by the
                             tot_weight to get the normalized objective function.
*/
void ComputeKLDObjectiveFunction(const GeneralMatrix &supervision,
                                 ObjectiveType objective_type,
                                 const std::string &output_name,
                                 bool supply_deriv,
                                 const float32 rho,
                                 NnetComputer *si_computer,
                                 NnetComputer *computer,
                                 BaseFloat *tot_weight,
                                 BaseFloat *tot_objf);


} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_KLD_TRAINING_H_
