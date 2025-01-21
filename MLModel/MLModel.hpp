#ifndef MLMODEL_HPP
#define MLMODEL_HPP

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/torch.h>

// TODO Specify kind of models and enumerate
// Basic kind I can think of
// 1. Core model: numberOfParticles, coord, neighbours count, neighbour list
// 2. Descriptor model: numberOfParticles, coord, neighbours count, neighbour
// list
// 3. Graph model: TBA
enum MLModelType {
  ML_MODEL_PYTORCH,
};

/* Abstract base class for an ML model -- 'product' of the factory pattern */
// Is this class even needed? Different ML model modes and libraries have really
// different APIs, and very difficult to contain in single abstract class
class MLModel
{
 public:
  static MLModel * create(const char * /*model_file_path*/,
                          MLModelType /*ml_model_type*/,
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

  virtual void Run(std::vector<torch::Tensor> &) = 0;

  virtual void SetInputSize(int) = 0;

  virtual ~MLModel() {};
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
    // try to always clone, managing lifetimes can be tricky.
    torch::TensorOptions options
        = torch::TensorOptions().device(*device_).requires_grad(requires_grad);
    if (std::is_same<T, int>::value)
    {
      options = options.dtype(torch::kInt32);  // default int is 32 bit?
    }
    else if (std::is_same<T, std::int64_t>::value)
    {
      options = options.dtype(torch::kInt64);
    }
    else if (std::is_same<T, double>::value)
    {
      options = options.dtype(torch::kFloat64);
    }
    else if ((std::is_same<T, double>::value
              && get_torch_default_floating_type() == torch::kFloat32)
             || std::is_same<T, float>::value)
    {
      options = options.dtype(torch::kFloat32);
    }
    else
    {
      throw std::runtime_error(
          "Invalid datatype provided as input to the model");
    }
    torch::Tensor input_tensor;
    // clone by default, avoid tricky memory lifetimes of tensors
    if (clone)
    {
      input_tensor = torch::from_blob(data, shape, options).clone();
    }
    else { input_tensor = torch::from_blob(data, shape, options); }

    // torch bug workaround
    if (requires_grad) { input_tensor.retain_grad(); }
    model_inputs_[idx] = std::move(input_tensor);
  }

 public:
  const char * model_file_path_;

  PytorchModel(const char * /*model_file_path*/,
               const char * /*device_name*/,
               int /*input size*/);

  void SetInputNode(int /*model_input_index*/,
                    double * /*input*/,
                    std::vector<std::int64_t> & /*arb size*/,
                    bool requires_grad,
                    bool clone);

  void SetInputNode(int /*model_input_index*/,
                    int * /*input*/,
                    std::vector<std::int64_t> & /*arb size*/,
                    bool requires_grad,
                    bool clone);

  void SetInputNode(int /*model_input_index*/,
                    std::int64_t * /*input*/,
                    std::vector<std::int64_t> & /*arb size*/,
                    bool requires_grad,
                    bool clone);

  void SetInputSize(int);

  //    void Run(double * /*energy*/, double * /*forces*/);
  void Run(std::vector<torch::Tensor> &);

  ~PytorchModel();
};

#endif /* MLMODEL_HPP */