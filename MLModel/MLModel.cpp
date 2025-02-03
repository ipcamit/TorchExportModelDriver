#include "MLModel.hpp"
#include <iostream>
#include <map>
#include <string>

// #ifdef USE_MPI
// #include <algorithm>
// #include <mpi.h>
// #include <unistd.h>
// #endif

#include <torch/script.h>

MLModel * MLModel::create(const char * model_file_path,
                          const char * const device_name,
                          const int model_input_size)
{
    return new PytorchModel(model_file_path, device_name, model_input_size);
}

void PytorchModel::SetExecutionDevice(const char * const device_name)
{
  device_ = new torch::Device("cpu");
}
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
  return torch::kFloat32; // smarter way to assign it? perhaps from model
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


void PytorchModel::Run(double * energy, double * partial_energy, double * forces)
{
  // FIXME: Make this work for arbitrary number/type of outputs?  This may
  // lead us to make Run() take no parameters, and instead define separate
  // methods for accessing each of the outputs of the ML model.

  // Run ML model's `forward` method and retrieve outputs as tuple
  // IMPORTANT: We require that the pytorch model's `forward`
  // method return a tuple where the energy is the first entry and
  // the forces are the second

  c10::InferenceMode();
  // std::cout << model_inputs_[0];
  std::cout << "Running model\n";
  //TEST

    // #include "/home/amit/Projects/COLABFIT/TorchExport/data_new.cpp"
    // auto pos_tensor = torch::from_blob(pos, pos_shape);
    // auto batch_tensor = torch::from_blob(batch, batch_shape);
    // auto natoms_tensor = torch::from_blob(natoms, natoms_shape);
    // auto atomic_numbers_tensor = torch::from_blob(atomic_numbers, atomic_numbers_shape);
    // auto edge_index_tensor = torch::from_blob(edge_index, edge_index_shape);
    // auto edge_distance_tensor = torch::from_blob(edge_distance, edge_distance_shape);
    // auto edge_distance_vec_tensor = torch::from_blob(edge_distance_vec, edge_distance_vec_shape);
    // std::vector<torch::Tensor> inputs;

    // inputs.push_back(pos_tensor);
    // inputs.push_back(batch_tensor);
    // inputs.push_back(natoms_tensor);
    // inputs.push_back(atomic_numbers_tensor);
    // inputs.push_back(edge_index_tensor);
    // inputs.push_back(edge_distance_tensor);
    // inputs.push_back(edge_distance_vec_tensor);

    // for (std::size_t i = 0; i < model_inputs_.size(); i++){
    //   std::cout << model_inputs_[i] << "\n";
    //
    // }

    std::cout << std::endl;

    std::vector<torch::Tensor> out_tensor;
    try
    {
      out_tensor = module_->run(model_inputs_);
    } catch (const c10::Error & e) {
      std::cerr << "PyTorch Error: " << e.what() << std::endl;
    }

  std::cout << "Ran model\n";
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

  if (partial_energy && (energy_tensor.numel() > 1)){
    std::memcpy(partial_energy, energy_tensor.contiguous().data_ptr<double>(), energy_tensor.numel());
  } else if (partial_energy && energy_tensor.numel() <= 1) {
    if (!warned_once_partial_energy){
      std::cerr << "===================================================" << std::endl;
      std::cerr << "PARTIAL ENERGY REQUESTED, BUT NOT PROVIDED BY MODEL" << std::endl;
      std::cerr << "===================================================" << std::endl;
      warned_once_partial_energy = true;
    }
  }



}

PytorchModel::PytorchModel(const char * model_file_path,
                           const char * device_name,
                           const int size_)
{
  model_file_path_ = model_file_path;
  SetExecutionDevice(device_name);
  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::cout << "Loading HARDCODED MODEL ESCN\n";
    module_ = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        "escn.so");
    std::cout << "LOADED\n";
  }
  catch (const c10::Error & e)
  {
    std::cerr << "ERROR: An error occurred while attempting to load the "
                 "pytorch model file from path "
              << model_file_path << std::endl;
    throw;
  }

  //    SetExecutionDevice(device_name); // Not needed yet

  // Copy model to execution device
  // module_.to(*device_);

  // Reserve size for the four fixed model inputs (particle_contributing,
  // coordinates, number_of_neighbors, neighbor_list)
  // Model inputs to be determined
  // model_inputs_.resize(size_); // Can use push_back now
  // SetInputSize(size_);

  // Set model to evaluation mode to set any dropout or batch normalization
  // layers to evaluation mode
  // module_.eval();
  // module_ = torch::jit::freeze(module_);

  // torch::jit::FusionStrategy strategy;
  // strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  // torch::jit::setFusionStrategy(strategy);
}

// void PytorchModel::GetInputNode(std::vector<torch::Tensor> &out_tensor) {
// return first tensor with grad = True
// for (auto &Ival: model_inputs_) {
//     if (Ival.toTensor().requires_grad()) {
//         out_tensor = Ival;
//         return;
//     }
// }
// }

// void PytorchModel::GetInputNode(int index, torch::Tensor &out_tensor) {
//     // return first tensor with grad = True
//     out_tensor = model_inputs_[index];
// }

void PytorchModel::SetInputSize(int size) {

  std::cout << "SETUP INPUT SIZE\n";
  model_inputs_.resize(size); }

PytorchModel::~PytorchModel() { delete device_; }