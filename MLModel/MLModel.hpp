#ifndef MLMODEL_HPP
#define MLMODEL_HPP

#include <cstdlib>
#include <memory>  // std::unique_ptr
#include <string>
#include <type_traits>  // std::is_same
#include <vector>

#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifndef CPU_ONLY  // exclude cuda runner if CPU only requested
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#include <torch/script.h>


// common datatype to torch type map.
template<typename T>
torch::Dtype getTorchDtype()
{
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

  virtual ~MLModel() = default;

  const std::string device_str;
};

/* Abstracted AOTInductor container for future proofing. It is rapidly changing
 * API.
 */
class AOTInductorModelContainer
{
 public:
  virtual std::vector<torch::Tensor> Run(std::vector<torch::Tensor> &) = 0;
  virtual ~AOTInductorModelContainer() = default;
  static std::unique_ptr<AOTInductorModelContainer>
  load_inductor_container(const std::string & so_path,
                          const std::string & device);
};

// This is how torch 2.4 was running the containers, but apparently it is
// changed now? TODO: Update it with respect to latest version. 2.6 onwards

class AOTInductorModelContainerCPU : public AOTInductorModelContainer
{
 private:
  torch::inductor::AOTIModelContainerRunnerCpu torch_module;

 public:
  std::vector<torch::Tensor> Run(std::vector<torch::Tensor> & inputs) override
  {
    return torch_module.run(inputs);
  }
  AOTInductorModelContainerCPU(const std::string & model_path) :
      torch_module(model_path) {};
};

class AOTInductorModelContainerGPU : public AOTInductorModelContainer
{
#ifndef CPU_ONLY
 private:
  torch::inductor::AOTIModelContainerRunnerCuda torch_module;

 public:
  AOTInductorModelContainerGPU(const std::string & model_path) :
      torch_module(model_path)
  {
  }

  std::vector<torch::Tensor> Run(std::vector<torch::Tensor> & inputs) override
  {
    return torch_module.run(inputs);
  }
#else
 public:
  AOTInductorModelContainerGPU(const std::string & model_path)
  {
    throw std::runtime_error(
        "Request to run GPU exported model, but the TorchExport driver was "
        "compiled with only CPU support (-DCPU_ONLY)");
  }

  std::vector<torch::Tensor> Run(std::vector<torch::Tensor> & inputs) override
  {
    throw std::runtime_error(
        "Run() called on GPU model, but compiled with CPU-only support.");
  }
#endif
};


// Concrete MLModel corresponding to pytorch
class PytorchModel : public MLModel
{
 private:
  std::unique_ptr<AOTInductorModelContainer> module_;
  int model_input_size;
  std::vector<torch::Tensor> model_inputs_;
  std::unique_ptr<torch::Device> device_;


  torch::Dtype get_torch_data_type(int *);
  torch::Dtype get_torch_default_floating_type();

  void SetExecutionDevice(const std::string & /*device_name*/);

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
  const std::string model_file_path_;
  const std::string device_str;

  PytorchModel(const std::string & /*model_file_path*/,
               const std::string & /*device_name*/,
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


  void Run(double * /*energy*/,
           double * /*partial energy*/,
           double * /*forces*/) override;

  ~PytorchModel() override;
};

#endif /* MLMODEL_HPP */
