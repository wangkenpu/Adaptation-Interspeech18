// nnet3/nnet-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2015    Xiaohui Zhang
//                2017    Ke Wang       Xiaomi Corporation

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

#include "nnet3/nnet-training-kld.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetKLDTrainer::NnetKLDTrainer(const NnetTrainerOptions &config,
                               const float32 rho, Nnet *si_nnet, Nnet *nnet):
    config_(config),
    nnet_(nnet),
    rho_(rho),
    si_nnet_(si_nnet),
    compiler_(*nnet_, config_.optimize_config, config_.compiler_config),
    si_compiler_(*si_nnet_, config_.optimize_config, config_.compiler_config),
    num_minibatches_processed_(0),
    srand_seed_(RandInt(0, 100000)) {
  if (config.zero_component_stats)
    ZeroComponentStats(nnet);
  KALDI_ASSERT(config.momentum >= 0.0 &&
               config.max_param_change >= 0.0);
  delta_nnet_ = nnet_->Copy();
  ScaleNnet(0.0, delta_nnet_);
  const int32 num_updatable = NumUpdatableComponents(*delta_nnet_);
  num_max_change_per_component_applied_.resize(num_updatable, 0);
  num_max_change_global_applied_ = 0;

  if (config_.read_cache != "") {
    bool binary;
    Input ki;
    if (ki.Open(config_.read_cache, &binary)) {
      compiler_.ReadCache(ki.Stream(), binary);
      KALDI_LOG << "Read computation cache from " << config_.read_cache;
    } else {
      KALDI_WARN << "Could not open cached computation. "
                    "Probably this is the first training iteration.";
    }
  }
}

void NnetKLDTrainer::Train(const NnetExample &eg) {
  bool need_model_derivative = true;
  ComputationRequest si_request, request;
  GetComputationRequest(*si_nnet_, eg, false,
                        false,
                        &si_request);
  GetComputationRequest(*nnet_, eg, need_model_derivative,
                        config_.store_component_stats,
                        &request);
  const NnetComputation *si_computation = si_compiler_.Compile(si_request);
  const NnetComputation *computation = compiler_.Compile(request);
  TrainInternal(eg, *si_computation, *computation);

  num_minibatches_processed_++;
}

void NnetKLDTrainer::TrainInternal(const NnetExample &eg,
                                   const NnetComputation &si_computation,
                                   const NnetComputation &computation) {
  NnetComputer si_computer(config_.compute_config, si_computation,
                           *si_nnet_, NULL);
  NnetComputer computer(config_.compute_config, computation,
                        *nnet_, delta_nnet_);
  // give the inputs to the computer object.
  si_computer.AcceptInputs(*si_nnet_, eg.io);
  si_computer.Run();
  computer.AcceptInputs(*nnet_, eg.io);
  computer.Run();

  this->ProcessOutputs(eg, &si_computer, &computer);
  computer.Run();

  // If relevant, add in the part of the gradient that comes from L2
  // regularization.
  ApplyL2Regularization(*nnet_,
                        GetNumNvalues(eg.io, false) * config_.l2_regularize_factor,
                        delta_nnet_);

  // Update the parameters of nnet
  bool success = UpdateNnetWithMaxChange(*delta_nnet_, config_.max_param_change,
      1.0, 1.0 - config_.momentum, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);

  // Scale down the batchnorm stats (keeps them fresh... this affects what
  // happens when we use the model with batchnorm test-mode set).
  ScaleBatchnormStats(config_.batchnorm_stats_scale, nnet_);

  // Scale deta_nnet
  if (success)
    ScaleNnet(config_.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);
}

void NnetKLDTrainer::ProcessOutputs(const NnetExample &eg,
                                    NnetComputer *si_computer,
                                    NnetComputer *computer) {
  // normally the eg will have just one output named 'output', but
  // we don't assume this.
  std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
      end = eg.io.end();
  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 si_node_index = si_nnet_->GetNodeIndex(io.name);
    int32 node_index = nnet_->GetNodeIndex(io.name);
    KALDI_ASSERT(si_node_index >= 0);
    KALDI_ASSERT(node_index >= 0);
    if (nnet_->IsOutputNode(node_index)) {
      ObjectiveType obj_type = nnet_->GetNode(node_index).u.objective_type;
      // Just support kLinear objective function.
      KALDI_ASSERT(obj_type == kLinear);
      BaseFloat tot_weight, tot_objf;
      bool supply_deriv = true;
      ComputeKLDObjectiveFunction(io.features, obj_type, io.name,
                                  supply_deriv, rho_, si_computer, computer,
                                  &tot_weight, &tot_objf);
      objf_info_[io.name].UpdateStats(io.name,
                                      config_.print_interval,
                                      num_minibatches_processed_,
                                      tot_weight, tot_objf);
    }
  }
}

bool NnetKLDTrainer::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  std::vector<std::pair<std::string, const ObjectiveFunctionInfo*> > all_pairs;
  for (; iter != end; ++iter)
    all_pairs.push_back(std::pair<std::string, const ObjectiveFunctionInfo*>(
        iter->first, &(iter->second)));
  // ensure deterministic order of these names (this will matter in situations
  // where a script greps for the objective from the log).
  std::sort(all_pairs.begin(), all_pairs.end());
  bool ans = false;
  for (size_t i = 0; i < all_pairs.size(); i++) {
    const std::string &name = all_pairs[i].first;
    const ObjectiveFunctionInfo &info = *(all_pairs[i].second);
    bool ok = info.PrintTotalStats(name);
    ans = ans || ok;
  }
  PrintMaxChangeStats();
  return ans;
}

void NnetKLDTrainer::PrintMaxChangeStats() const {
  KALDI_ASSERT(delta_nnet_ != NULL);
  int32 i = 0;
  for (int32 c = 0; c < delta_nnet_->NumComponents(); c++) {
    Component *comp = delta_nnet_->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
                  << "UpdatableComponent; change this code.";
      if (num_max_change_per_component_applied_[i] > 0)
        KALDI_LOG << "For " << delta_nnet_->GetComponentName(c)
                  << ", per-component max-change was enforced "
                  << (100.0 * num_max_change_per_component_applied_[i]) /
                     (num_minibatches_processed_ *
                     (config_.backstitch_training_scale == 0.0 ? 1.0 :
                     1.0 + 1.0 / config_.backstitch_training_interval))
                  << " \% of the time.";
      i++;
    }
  }
  if (num_max_change_global_applied_ > 0)
    KALDI_LOG << "The global max-change was enforced "
              << (100.0 * num_max_change_global_applied_) /
                 (num_minibatches_processed_ *
                 (config_.backstitch_training_scale == 0.0 ? 1.0 :
                 1.0 + 1.0 / config_.backstitch_training_interval))
              << " \% of the time.";
}

NnetKLDTrainer::~NnetKLDTrainer() {
  if (config_.write_cache != "") {
    Output ko(config_.write_cache, config_.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), config_.binary_write_cache);
    KALDI_LOG << "Wrote computation cache to " << config_.write_cache;
  }
  delete delta_nnet_;
}

void ComputeKLDObjectiveFunction(const GeneralMatrix &supervision,
                                 ObjectiveType objective_type,
                                 const std::string &output_name,
                                 bool supply_deriv,
                                 const float32 rho,
                                 NnetComputer *si_computer,
                                 NnetComputer *computer,
                                 BaseFloat *tot_weight,
                                 BaseFloat *tot_objf) {
  const CuMatrixBase<BaseFloat> &si_output = si_computer->GetOutput(output_name);
  CuMatrix<BaseFloat> si_post(si_output);
  si_post.ApplyExp();
  si_post.Scale(rho);
  const CuMatrixBase<BaseFloat> &output = computer->GetOutput(output_name);

  if (output.NumCols() != supervision.NumCols())
    KALDI_ERR << "Nnet versus example output dimension (num-classes) "
              << "mismatch for '" << output_name << "': " << output.NumCols()
              << " (nnet) vs. " << supervision.NumCols() << " (egs)\n";

  switch (objective_type) {
    case kLinear: {
      // objective is x * y.
      switch (supervision.Type()) {
        case kSparseMatrix: {
          const SparseMatrix<BaseFloat> &post = supervision.GetSparseMatrix();
          CuSparseMatrix<BaseFloat> cu_post(post);
          // The cross-entropy objective is computed by a simple dot product,
          // because after the LogSoftmaxLayer, the output is already in the form
          // of log-likelihoods that are normalized to sum to one.
          *tot_weight = cu_post.Sum();
          CuMatrix<BaseFloat> cu_kld_post(output.NumRows(), output.NumCols(),
                                          kUndefined);
          cu_post.CopyToMat(&cu_kld_post);
          cu_kld_post.Scale(1.0 - rho);
          cu_kld_post.AddMat(1.0, si_post);
          *tot_objf = TraceMatMat(output, cu_kld_post, kTrans);
          if (supply_deriv) {
            computer->AcceptInput(output_name, &cu_kld_post);
          }
          break;
        }
        case kFullMatrix: {
          // there is a redundant matrix copy in here if we're not using a GPU
          // but we don't anticipate this code branch being used in many cases.
          CuMatrix<BaseFloat> cu_post(supervision.GetFullMatrix());
          *tot_weight = cu_post.Sum();
          cu_post.Scale(1.0 - rho);
          cu_post.AddMat(1.0, si_post);
          *tot_objf = TraceMatMat(output, cu_post, kTrans);
          if (supply_deriv)
            computer->AcceptInput(output_name, &cu_post);
          break;
        }
        case kCompressedMatrix: {
          Matrix<BaseFloat> post;
          supervision.GetMatrix(&post);
          CuMatrix<BaseFloat> cu_post;
          cu_post.Swap(&post);
          *tot_weight = cu_post.Sum();
          cu_post.Scale(1.0 - rho);
          cu_post.AddMat(1.0, si_post);
          *tot_objf = TraceMatMat(output, cu_post, kTrans);
          if (supply_deriv)
            computer->AcceptInput(output_name, &cu_post);
          break;
        }
      }
      break;
    }
    case kQuadratic: {
      // objective is -0.5 (x - y)^2
      // Don't support currently.
      KALDI_ERR << "Do not support Quatratic objetct function.";
      CuMatrix<BaseFloat> diff(supervision.NumRows(),
                               supervision.NumCols(),
                               kUndefined);
      diff.CopyFromGeneralMat(supervision);
      diff.AddMat(-1.0, output);
      *tot_weight = diff.NumRows();
      *tot_objf = -0.5 * TraceMatMat(diff, diff, kTrans);
      if (supply_deriv)
        computer->AcceptInput(output_name, &diff);
      break;
    }
    default:
      KALDI_ERR << "Objective function type " << objective_type
                << " not handled.";
  }
}



} // namespace nnet3
} // namespace kaldi
