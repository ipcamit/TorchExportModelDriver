#ifndef MLMODEL_HPP
#define MLMODEL_HPP

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/script.h>


// common datatype to torch type map.
template <typename T>
torch::Dtype getTorchDtype() {
    if (std::is_same<T, int>::value) return torch::kInt32;
    if (std::is_same<T, std::int64_t>::value) return torch::kInt64;
    if (std::is_same<T, float>::value) return torch::kFloat32;
    if (std::is_same<T, double>::value) return torch::kFloat64;
    throw std::runtime_error("Invalid datatype provided as input to the model");
}


/* Abstract base class for an ML model -- 'product' of the factory pattern */
// Is this class even needed? Different ML model modes and libraries have really
// different APIs, and very difficult to contain in single abstract class
class MLModel
{
 public:
  static MLModel * create(const char * /*model_file_path*/,
                          const char * /*device_name*/,
                          int /*model_input_size*/);

  // TODO: Should we use named inputs instead?  I believe they're required
  // by ONNX, but not sure exactly how they work vis-a-vis exporting to a
  // torchscript file.

  // Function templates can't be used for pure virtual functions, and since
  // SetInputNode and Run each have their own (different) support argument
  // types, we can't use a class template.  So, we explicitly define each
  // supported overloading.
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

  virtual void Run(double *, double *, double *) = 0;

  virtual void SetInputSize(int) = 0;

  virtual ~MLModel() = default;
};

// Concrete MLModel corresponding to pytorch
class PytorchModel : public MLModel
{
 private:
  std::unique_ptr<torch::inductor::AOTIModelContainerRunnerCpu> module_;
  std::vector<torch::Tensor> model_inputs_;
  torch::Device * device_;  // TODO: Is this needed? Now CPU and GPU models are
                            // differently loaded

  torch::Dtype get_torch_data_type(int *);
  torch::Dtype get_torch_default_floating_type();

  void SetExecutionDevice(const char * /*device_name*/);

template<typename T>
void SetInputNodeTemplate(int idx,
                          T * data,
                          std::vector<std::int64_t> & shape,
                          bool requires_grad,
                          bool clone)
{
  // Configure tensor options
  torch::TensorOptions options = torch::TensorOptions()
                                     .device(*device_)
                                     .dtype(getTorchDtype<T>())
                                     .requires_grad(requires_grad);

  // Create tensor (clone if necessary)
  torch::Tensor input_tensor = torch::from_blob(data, shape, options);
  if (clone) input_tensor = input_tensor.clone();

  // Convert floating-point types if necessary
  if ((std::is_floating_point<T>::value)
      && (input_tensor.dtype() != get_torch_default_floating_type()))
  {
    input_tensor = input_tensor.to(get_torch_default_floating_type());
  }

  // Workaround for PyTorch bug
  if (requires_grad) input_tensor.retain_grad();

  model_inputs_.push_back(input_tensor);
}

bool warned_once_partial_energy = false;

 public:
  const char * model_file_path_;

  PytorchModel(const char * /*model_file_path*/,
               const char * /*device_name*/,
               int /*input size*/);

  void SetInputNode(int /*model_input_index*/,
                    double * /*input*/,
                    std::vector<std::int64_t> & /*arb size*/,
                    bool requires_grad,
                    bool clone) override;

  void SetInputNode(int /*model_input_index*/,
                    int * /*input*/,
                    std::vector<std::int64_t> & /*arb size*/,
                    bool requires_grad,
                    bool clone) override;

  void SetInputNode(int /*model_input_index*/,
                    std::int64_t * /*input*/,
                    std::vector<std::int64_t> & /*arb size*/,
                    bool requires_grad,
                    bool clone) override;

  void SetInputSize(int) override;

  void Run(double */*energy*/, double */*partial energy*/, double */*forces*/) override;

  ~PytorchModel() override;
};

#endif /* MLMODEL_HPP */
