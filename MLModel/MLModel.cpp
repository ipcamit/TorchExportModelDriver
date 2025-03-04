#include "MLModel.hpp"
#include <iostream>
#include <map>
#include <string>

// #ifdef USE_MPI
// #include <algorithm>
// #include <mpi.h>
// #include <unistd.h>
// #endif

MLModel * MLModel::create(const char * model_file_path,
                          const char * device_name,
                          const int model_input_size)
{
  return new PytorchModel(model_file_path, device_name, model_input_size);
}

void PytorchModel::SetExecutionDevice(const std::string & device_name)
{
  if (!device_name.empty())
    device_ = std::make_unique<torch::Device>(device_name);
  else { device_ = std::make_unique<torch::Device>("cpu"); }
}

/*
 * This is not used right now, but leaving it commented out for now, as in
 * future it will be needed for proper device allocation on multi GPU platform.
 */
//    // Use the requested device name char array to create a torch Device
//    // object.  Generally, the ``device_name`` parameter is going to come
//    // from a call to std::getenv(), so it is defined as const.
//
//    std::string device_name_as_str;
//
//    // Default to 'cpu'
//    if (device_name == nullptr) {
//        device_name_as_str = "cpu";
//    } else {
//        device_name_as_str = device_name;
//
//        //Only compile if MPI is detected
//        //n devices for n ranks, it will crash if MPI != GPU
//        // TODO: Add a check if GPU aware MPI can be used
//        #ifdef USE_MPI
//        std::cout << "INFO: Using MPI aware GPU allocation" << std::endl;
//        int rank=0, size = 0;
//        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//        MPI_Comm_size(MPI_COMM_WORLD, &size);
//        // get number of cuda devices visible
//        auto cuda_device_visible_env_var =
//        std::getenv("CUDA_VISIBLE_DEVICES"); //input "0,1,2"
//        std::vector<std::string> cuda_device_visible_ids;
//        int num_cuda_devices_visible = 0;
//        if (cuda_device_visible_env_var != nullptr){
//            std::string
//            cuda_device_visible_env_var_str(cuda_device_visible_env_var);
//            num_cuda_devices_visible =
//            std::count(cuda_device_visible_env_var_str.begin(),
//            cuda_device_visible_env_var_str.end(), ',') + 1; for (int i = 0; i
//            < num_cuda_devices_visible; i++) {
//                cuda_device_visible_ids.push_back(cuda_device_visible_env_var_str.substr(0,
//                cuda_device_visible_env_var_str.find(',')));
//                cuda_device_visible_env_var_str.erase(0,
//                cuda_device_visible_env_var_str.find(',') + 1);
//            }
//        } else {
//            throw std::invalid_argument("CUDA_VISIBLE_DEVICES not set\n "
//                                        "You requested for manual MPI aware
//                                        device allocation but
//                                        CUDA_VISIBLE_DEVICES is not set\n");
//        }
//        // assign cuda device to ranks in round-robin fashion
//        device_name_as_str += ":";
//        device_name_as_str += cuda_device_visible_ids[rank %
//        num_cuda_devices_visible]; char hostname[256]; gethostname(hostname,
//        256);
//        // poor man's sync print
//        for (int i = 0; i < size; i++) {
//            MPI_Barrier(MPI_COMM_WORLD);
//            if (i == rank) {
//                std::cout << "INFO: Rank " << rank << " on " << hostname << "
//                is using device " << device_name_as_str << std::endl;
//            }
//            MPI_Barrier(MPI_COMM_WORLD);
//        }
//
//        // auto kim_model_mpi_aware_env_var =
//        std::getenv("KIM_MODEL_MPI_AWARE");
//        // if ((kim_model_mpi_aware_env_var != NULL) &&
//        (strcmp(kim_model_mpi_aware_env_var, "yes") == 0)){
//        //     device_name_as_str += ":";
//        //     device_name_as_str += std::to_string(rank);
//        // }
//        #endif
//    }
//    device_ = new torch::Device(device_name_as_str);
//}


torch::Dtype PytorchModel::get_torch_data_type(int *)
{
  // Get the size used by 'int' on this platform and set torch tensor type
  // appropriately
  const std::size_t platform_size_int = sizeof(int);

  std::map<int, torch::Dtype> platform_size_int_to_torch_dtype;

  platform_size_int_to_torch_dtype[1] = torch::kInt8;
  platform_size_int_to_torch_dtype[2] = torch::kInt16;
  platform_size_int_to_torch_dtype[4] = torch::kInt32;
  platform_size_int_to_torch_dtype[8] = torch::kInt64;

  torch::Dtype torch_dtype
      = platform_size_int_to_torch_dtype[platform_size_int];

  return torch_dtype;
}

torch::Dtype PytorchModel::get_torch_default_floating_type()
{
  return torch::kFloat32;  // smarter way to assign it? perhaps from model
}

void PytorchModel::SetInputNode(int idx,
                                double * data,
                                std::vector<std::int64_t> & shape,
                                bool requires_grad = false,
                                bool clone = true)
{
  SetInputNodeTemplate(idx, data, shape, requires_grad, clone);
}

void PytorchModel::SetInputNode(int idx,
                                int * data,
                                std::vector<std::int64_t> & shape,
                                bool requires_grad = false,
                                bool clone = true)
{
  SetInputNodeTemplate(idx, data, shape, requires_grad, clone);
}

void PytorchModel::SetInputNode(int idx,
                                std::int64_t * data,
                                std::vector<std::int64_t> & shape,
                                bool requires_grad = false,
                                bool clone = true)
{
  SetInputNodeTemplate(idx, data, shape, requires_grad, clone);
}


void PytorchModel::Run(double * energy,
                       double * partial_energy,
                       double * forces)
{
  c10::InferenceMode guard(true);
  std::vector<torch::Tensor> out_tensor;

  try
  {
    out_tensor = module_->Run(model_inputs_);
  }
  catch (const c10::Error & e)
  {
    std::cerr << "PyTorch Error: " << e.what() << std::endl;
  }

  auto energy_tensor = out_tensor[0].to(torch::kCPU);
  auto torch_forces = out_tensor[1].to(torch::kCPU);

  std::cout << energy_tensor;

  energy_tensor = energy_tensor.to(torch::kFloat64);
  torch_forces = torch_forces.to(torch::kFloat64);

  auto force_size = torch_forces.numel();

  if (forces)
  {
    auto force_accessor = torch_forces.contiguous().data_ptr<double>();
    std::memcpy(forces, force_accessor, force_size);
  }

  if (energy)
  {
    *energy = *(energy_tensor.sum().contiguous().data_ptr<double>());
  }

  if (partial_energy && (energy_tensor.numel() > 1))
  {
    std::memcpy(partial_energy,
                energy_tensor.contiguous().data_ptr<double>(),
                energy_tensor.numel());
  }
  else if (partial_energy && energy_tensor.numel() <= 1)
  {
    if (!warned_once_partial_energy)
    {
      std::cerr << "==================================================="
                << std::endl;
      std::cerr << "PARTIAL ENERGY REQUESTED, BUT NOT PROVIDED BY MODEL"
                << std::endl;
      std::cerr << "==================================================="
                << std::endl;
      warned_once_partial_energy = true;
    }
  }

  model_inputs_.clear();
  model_inputs_.reserve(model_input_size);
}

PytorchModel::PytorchModel(const std::string & model_file_path,
                           const std::string & device_name,
                           const int size_) :
    model_input_size(size_), model_file_path_(model_file_path)
{
  SetExecutionDevice(device_name);
  try
  {
    module_ = AOTInductorModelContainer::load_inductor_container(
        model_file_path, device_name);
  }
  catch (const c10::Error & e)
  {
    std::string err("ERROR: An error occurred while attempting to load the "
                    "pytorch model file from path "
                    + model_file_path);
    throw std::runtime_error(err);
  }
}


PytorchModel::~PytorchModel() = default;
std::unique_ptr<AOTInductorModelContainer>
AOTInductorModelContainer::load_inductor_container(const std::string & so_path,
                                                   const std::string & device)
{
  if (device == "cpu")
    return std::make_unique<AOTInductorModelContainerCPU>(so_path);
  else if (device == "cuda")
    return std::make_unique<AOTInductorModelContainerGPU>(so_path);
  else
    throw std::runtime_error("Device: " + device + " not supported");
}
