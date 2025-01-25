//
// Created by amit on 7/12/22.
//

#ifndef TORCH_EXPORT_MODEL_DRIVER_IMPLEMENTATION_HPP
#define TORCH_EXPORT_MODEL_DRIVER_IMPLEMENTATION_HPP

#include "KIM_ModelDriverHeaders.hpp"
#include <array>
#include <set>
#include <vector>
// #include "MLModel.hpp"
// #include <torch/torch.h>

class MLModel;

class TorchExportModelDriverImplementation
{
 public:
  // All file params are public
  double influence_distance, cutoff_distance;
  int n_elements, n_layers;
  std::vector<std::string> elements_list;
  std::string preprocessing;
  std::string model_name;
  bool returns_forces;
  std::string descriptor_name = "None";
  std::string descriptor_param_file = "None";

  TorchExportModelDriverImplementation(
      KIM::ModelDriverCreate * modelDriverCreate,
      KIM::LengthUnit requestedLengthUnit,
      KIM::EnergyUnit requestedEnergyUnit,
      KIM::ChargeUnit requestedChargeUnit,
      KIM::TemperatureUnit requestedTemperatureUnit,
      KIM::TimeUnit requestedTimeUnit,
      int * ier);

  ~TorchExportModelDriverImplementation();

  int Refresh(KIM::ModelRefresh * modelRefresh);
  int Refresh(KIM::ModelDriverCreate * modelRefresh);

  int Compute(KIM::ModelComputeArguments const * modelComputeArguments);

  int ComputeArgumentsCreate(
      KIM::ModelComputeArgumentsCreate * modelComputeArgumentsCreate);

  int ComputeArgumentsDestroy(
      KIM::ModelComputeArgumentsDestroy * modelComputeArgumentsDestroy);

 private:
  // Derived or assigned variables are private
  int modelWillNotRequestNeighborsOfNoncontributingParticles_;
  int n_contributing_atoms;
  int number_of_inputs;
  int64_t * species_atomic_number;
  int64_t * contraction_array;

  MLModel * ml_model;

  std::vector<int> num_neighbors_;
  std::vector<int> neighbor_list;
  std::vector<int> z_map;

  double * descriptor_array;
  long ** graph_edge_indices;

  void
  updateNeighborList(KIM::ModelComputeArguments const * modelComputeArguments,
                     int numberOfParticles);

  void
  setDefaultInputs(const KIM::ModelComputeArguments * modelComputeArguments);

  // void setDescriptorInputs(const KIM::ModelComputeArguments
  // *modelComputeArguments);

  void setGraphInputs(const KIM::ModelComputeArguments * modelComputeArguments);

  void readParametersFile(KIM::ModelDriverCreate * modelDriverCreate,
                          int * ier);

  static void unitConversion(KIM::ModelDriverCreate * modelDriverCreate,
                             KIM::LengthUnit requestedLengthUnit,
                             KIM::EnergyUnit requestedEnergyUnit,
                             KIM::ChargeUnit requestedChargeUnit,
                             KIM::TemperatureUnit requestedTemperatureUnit,
                             KIM::TimeUnit requestedTimeUnit,
                             int * ier);

  void setSpecies(KIM::ModelDriverCreate * modelDriverCreate, int * ier);

  static void
  registerFunctionPointers(KIM::ModelDriverCreate * modelDriverCreate,
                           int * ier);

  void
  preprocessInputs(KIM::ModelComputeArguments const * modelComputeArguments);

  // void postprocessOutputs(std::vector<torch::Tensor> &,
  //                         KIM::ModelComputeArguments const *);

  void Run(KIM::ModelComputeArguments const * modelComputeArguments);

  void contributingAtomCounts(
      KIM::ModelComputeArguments const * modelComputeArguments);

  void graphSetToGraphArray(std::vector<std::set<std::tuple<long, long> > > &);
  // TorchMLModelImplementation * implementation_;
};

int sym_to_z(std::string &);

// For hashing unordered_set of pairs
// https://arxiv.org/pdf/2105.10752.pdf
class SymmetricCantorPairing
{
 public:
  int64_t operator()(const std::array<long, 2> & t) const
  {
    int64_t k1 = t[0];
    int64_t k2 = t[1];
    int64_t kmin = std::min(k1, k2);
    int64_t ksum = k1 + k2 + 1;

    return ((ksum * ksum - ksum % 2) + kmin) / 4;
  }
};

// Forward declaration of MLModel Class
// This is needed as we don't want to leak any dependency from the ML model
// to the model driver code. Sometime torch libraries interfere with the
// compilation of the model driver.
// Now ML model is completely isolated

class MLModel
{
 public:
  static MLModel * create(const char * /*model_file_path*/,
                          const char * /*device_name*/,
                          int /*model_input_size*/);

  virtual void SetInputNode(int /*model_input_index*/,
                            double * /*input*/,
                            std::vector<std::int64_t> & /*arb size*/,
                            bool requires_grad,
                            bool clone)
      = 0;

  virtual void SetInputNode(int /*model_input_index*/,
                            int * /*input*/,
                            std::vector<std::int64_t> & /*arb size*/,
                            bool requires_grad,
                            bool clone)
      = 0;

  virtual void SetInputNode(int /*model_input_index*/,
                            std::int64_t * /*input*/,
                            std::vector<std::int64_t> & /*arb size*/,
                            bool requires_grad,
                            bool clone)
      = 0;

  virtual void
  Run(double * /*energy*/, double * /*partial energy*/, double * /*forces*/)
      = 0;

  virtual void SetInputSize(int) = 0;

  virtual ~MLModel() = default;
};


#endif  // TORCH_EXPORT_MODEL_DRIVER_IMPLEMENTATION_HPP