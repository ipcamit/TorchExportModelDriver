#include "TorchExportModelDriverImplementation.hpp"
#include "KIM_LogMacros.hpp"
#include "TorchExportModelDriver.hpp"
#include "yaml.hpp"
#include <stdexcept>
#include <vector>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_set>

#define MAX_FILE_NUM 3
// typedef double VecOfSize3[3];
#define KIM_ELEMENTS_ENV_VAR "KIM_MODEL_ELEMENTS_MAP"
#define KIM_SO_ENV_VAR "KIM_MODEL_SO_PATH"
//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

TorchExportModelDriverImplementation::TorchExportModelDriverImplementation(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit,
    int * const ier)
{
  *ier = false;
  // initialize members to remove warning----
  std::cout << "<<EXPERIMENTAL TORCHEXPORT DRIVER>>\n";
  influence_distance = 0.0;
  n_elements = 0;
  ml_model = nullptr;
  cutoff_distance = 0.0;
  n_layers = 0;
  n_contributing_atoms = 0;
  number_of_inputs = 0;
  // Read parameter files from model driver
  // --------------------------------------- also initialize the ml_model
  readParametersFile(modelDriverCreate, ier);
  LOG_DEBUG("Read Param files");
  if (*ier) return;

  // Unit conversions
  // -----------------------------------------------------------------
  unitConversion(modelDriverCreate,
                 requestedLengthUnit,
                 requestedEnergyUnit,
                 requestedChargeUnit,
                 requestedTemperatureUnit,
                 requestedTimeUnit,
                 ier);
  LOG_DEBUG("Registered Unit Conversion");
  if (*ier) return;

  // Set Influence distance
  // ---------------------------------------------------------
  modelWillNotRequestNeighborsOfNoncontributingParticles_
      = static_cast<int>(true);

  modelDriverCreate->SetInfluenceDistancePointer(&influence_distance);
  modelDriverCreate->SetNeighborListPointers(
      1,
      &cutoff_distance,
      &modelWillNotRequestNeighborsOfNoncontributingParticles_);

  // Species code
  // --------------------------------------------------------------------
  setSpecies(modelDriverCreate, ier);
  LOG_DEBUG("Registered Species");
  if (*ier) return;

  // Register Index
  // settings-----------------------------------------------------------
  modelDriverCreate->SetModelNumbering(KIM::NUMBERING::zeroBased);

  // Register Parameters
  // -------------------------------------------------------------- All model
  // parameters are inside model So nothing to do here? Just Registering
  // n_elements lest KIM complaints. TODO Discuss with Ryan?
  *ier = modelDriverCreate->SetParameterPointer(
      1, &n_elements, "n_elements", "Number of elements");
  LOG_DEBUG("Registered Parameter");
  if (*ier) return;

  // Register function pointers
  // -----------------------------------------------------------
  registerFunctionPointers(modelDriverCreate, ier);
  if (*ier) return;

  // Set
  // Units----------------------------------------------------------------------------
  *ier = modelDriverCreate->SetUnits(requestedLengthUnit,
                                     requestedEnergyUnit,
                                     KIM::CHARGE_UNIT::unused,
                                     KIM::TEMPERATURE_UNIT::unused,
                                     KIM::TIME_UNIT::unused);

  // Set preprocessor descriptor callbacks
  // --------------------------------------------------
  graph_edge_indices = nullptr;
  species_atomic_number = nullptr;
  contraction_array = nullptr;
}

//******************************************************************************
// TODO: Can be done with templating. Deal with it later
int TorchExportModelDriverImplementation::Refresh(
    KIM::ModelRefresh * const modelRefresh)
{
  modelRefresh->SetInfluenceDistancePointer(&influence_distance);
  modelRefresh->SetNeighborListPointers(
      1,
      &cutoff_distance,
      &modelWillNotRequestNeighborsOfNoncontributingParticles_);
  // As all param are part of torch model, nothing to do here?
  // TODO Distance matrix for computational efficiency, which will be refreshed
  // to -1
  return false;
}

int TorchExportModelDriverImplementation::Refresh(
    KIM::ModelDriverCreate * const modelRefresh)
{
  modelRefresh->SetInfluenceDistancePointer(&influence_distance);
  modelRefresh->SetNeighborListPointers(
      1,
      &cutoff_distance,
      &modelWillNotRequestNeighborsOfNoncontributingParticles_);
  // TODO Distance matrix for computational efficiency, which will be refreshed
  // to -1
  return false;
}

//******************************************************************************
int TorchExportModelDriverImplementation::Compute(
    KIM::ModelComputeArguments const * const modelComputeArguments)
{
  Run(modelComputeArguments);
  // TODO see proper way to return error codes

  return false;
}

//*****************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArgumentsCreate

int TorchExportModelDriverImplementation::ComputeArgumentsCreate(
    KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate)
{
  // LOG_INFORMATION("Compute argument create");
  int error = modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
                  KIM::SUPPORT_STATUS::required)
              // Support state can be optional for energy
              // But then we need to explicitly handle nullptr case
              // As energy is anyway always needs to be computed
              // as a prerequisite for force computation, easier to
              // make it required.
              // TODO: Handle it properly in future for GD models
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialForces,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialParticleEnergy,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
                  KIM::SUPPORT_STATUS::notSupported);
  // register callbacks
  error = error
          || modelComputeArgumentsCreate->SetCallbackSupportStatus(
              KIM::COMPUTE_CALLBACK_NAME::ProcessDEDrTerm,
              KIM::SUPPORT_STATUS::notSupported)
          || modelComputeArgumentsCreate->SetCallbackSupportStatus(
              KIM::COMPUTE_CALLBACK_NAME::ProcessD2EDr2Term,
              KIM::SUPPORT_STATUS::notSupported);
  return error;
  LOG_INFORMATION("Register callback supportStatus");
}

// *****************************************************************************
// Auxiliary methods------------------------------------------------------------

void TorchExportModelDriverImplementation::Run(
    const KIM::ModelComputeArguments * const modelComputeArguments)
{
  contributingAtomCounts(modelComputeArguments);
  preprocessInputs(modelComputeArguments);

  double * energy = nullptr;
  double * partialEnergy = nullptr;
  double * forces = nullptr;
  int * particleSpeciesCodes;  // FIXME: Implement species code handling
  int const * numberOfParticlesPointer;
  int const * particleContributing = nullptr;
  double * coordinates = nullptr;
  // double *virial = nullptr; // Not supported yet

  auto ier = modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::partialForces,
                 static_cast<double **>(&forces))
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::partialEnergy, &energy);
  //  || modelComputeArguments->GetArgumentPointer(
  //   KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
  //   &virial);
  if (ier) return;

  ml_model->Run(energy, partialEnergy, forces);
}

// -----------------------------------------------------------------------------
void TorchExportModelDriverImplementation::preprocessInputs(
    KIM::ModelComputeArguments const * const modelComputeArguments)
{
  // TODO: Make preprocessing type enums
  if (preprocessing == "none") { setDefaultInputs(modelComputeArguments); }
  else if (preprocessing == "graph") { setGraphInputs(modelComputeArguments); }
}


// -----------------------------------------------------------------------------
void TorchExportModelDriverImplementation::updateNeighborList(
    KIM::ModelComputeArguments const * const modelComputeArguments,
    int const numberOfParticles)
{
  int const * numberOfParticlesPointer = nullptr;
  int const * particleContributing = nullptr;
  auto ier = modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
                 &numberOfParticlesPointer)
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
                 &particleContributing);
  if (ier)
  {
    // LOG_ERROR(
    //     "Could not create model compute arguments input @
    //     updateNeighborList");
    return;
  }
  int numOfNeighbors;
  int const * neighbors;
  num_neighbors_.clear();
  neighbor_list.clear();
  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    if (particleContributing[i] == 1)
    {
      modelComputeArguments->GetNeighborList(0, i, &numOfNeighbors, &neighbors);
      num_neighbors_.push_back(numOfNeighbors);
      for (int neigh = 0; neigh < numOfNeighbors; neigh++)
      {
        neighbor_list.push_back(neighbors[neigh]);
      }
    }
  }
}

// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchExportModelDriverImplementation::setDefaultInputs(
    const KIM::ModelComputeArguments * modelComputeArguments)
{
  int const * numberOfParticlesPointer;
  int *
      particleSpeciesCodes;  // FIXME: Implement species code handling Ask Ryan
  int * particleContributing = nullptr;
  double * coordinates = nullptr;

  auto ier = modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
                 &numberOfParticlesPointer)
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
                 &particleSpeciesCodes)
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
                 &particleContributing)
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::coordinates,
                 const_cast<double const **>(&coordinates));
  if (ier)
  {
    LOG_ERROR(
        "Could not create model compute arguments input @ setDefaultInputs");
    return;
  }

  if (species_atomic_number)
  {
    delete[] species_atomic_number;
    species_atomic_number = nullptr;
  }

  bool map_species_z = std::getenv(KIM_ELEMENTS_ENV_VAR) != nullptr;

  species_atomic_number = new int64_t[*numberOfParticlesPointer];
  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    if (map_species_z)
    {
      species_atomic_number[i] = z_map[particleSpeciesCodes[i]];
    }
    else { species_atomic_number[i] = particleSpeciesCodes[i]; }
  }


  int n_particles = *numberOfParticlesPointer;

  std::vector<std::int64_t> shape = {static_cast<std::int64_t>(n_particles)};

  ml_model->SetInputNode(0, species_atomic_number, shape, false, true);


  shape.clear();
  shape = {*numberOfParticlesPointer, 3};

  ml_model->SetInputNode(1, coordinates, shape, true, true);

  updateNeighborList(modelComputeArguments, n_contributing_atoms);

  shape.clear();
  shape = {static_cast<std::int64_t>(num_neighbors_.size())};
  ml_model->SetInputNode(2, num_neighbors_.data(), shape, false, true);

  shape.clear();
  shape = {static_cast<std::int64_t>(neighbor_list.size())};
  ml_model->SetInputNode(3, neighbor_list.data(), shape, false, true);


  if (contraction_array)
  {
    delete[] contraction_array;
    contraction_array = nullptr;
  }


  contraction_array = new int64_t[*numberOfParticlesPointer];
  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    contraction_array[i] = particleContributing[i];
  }
  shape.clear();
  shape = {*numberOfParticlesPointer};
  ml_model->SetInputNode(4, contraction_array, shape, false, true);
}


// -----------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchExportModelDriverImplementation::setGraphInputs(
    const KIM::ModelComputeArguments * modelComputeArguments)
{
  int const * numberOfParticlesPointer;
  int * particleSpeciesCodes;  // FIXME: Implement species code handling
  int * particleContributing = nullptr;
  double * coordinates = nullptr;
  auto ier = modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
                 &numberOfParticlesPointer)
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
                 &particleSpeciesCodes)
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
                 &particleContributing)
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::coordinates,
                 const_cast<double const **>(&coordinates));
  if (ier)
  {
    LOG_ERROR(
        "Could not create model compute arguments input @ setDefaultInputs");
    return;
  }
  int numberOfNeighbors;
  int const * neighbors;

  std::vector<int> atoms_in_layers;
  atoms_in_layers.reserve(*numberOfParticlesPointer);
  std::
      unordered_set<std::array<long, 2>, SymmetricCantorPairing, SymmetricEqual>
          graph;

  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    if (particleContributing[i] == 1) { atoms_in_layers.push_back(i); }
  }

  double cutoff_sq = cutoff_distance * cutoff_distance;
  double _x, _y, _z;
  std::array<double, 3> i_arr = {0.0, 0.0, 0.0}, j_arr = {0.0, 0.0, 0.0};
  std::unordered_set<int> atoms_in_next_layer;

  // TODO: This is now trivially parallalizable
  for (auto atom_i : atoms_in_layers)
  {
    modelComputeArguments->GetNeighborList(
        0, atom_i, &numberOfNeighbors, &neighbors);
    for (int j = 0; j < numberOfNeighbors; j++)
    {
      int atom_j = neighbors[j];
      std::memcpy(i_arr.data(), coordinates + 3 * atom_i, 3 * sizeof(double));
      std::memcpy(j_arr.data(), coordinates + 3 * atom_j, 3 * sizeof(double));
      _x = j_arr[0] - i_arr[0];
      _y = j_arr[1] - i_arr[1];
      _z = j_arr[2] - i_arr[2];
      double r_sq = _x * _x + _y * _y + _z * _z;
      if (r_sq <= cutoff_sq) { graph.insert({atom_i, atom_j}); }
    }
  }

  if (species_atomic_number)
  {
    delete[] species_atomic_number;
    species_atomic_number = nullptr;
  }

  // TODO: Read this from file
  //  Get environment variable KIM_MODEL_ELEMENTS_MAP, and set map_species_z to
  //  true if it is set, else false
  bool map_species_z = std::getenv(KIM_ELEMENTS_ENV_VAR) != nullptr;

  // TODO: benchmark smart pointers. Using new for performance and persistence
  // Memory ownership in torch blobs is tricky for CPU. Will be deep copied
  // anyway in GPU.
  species_atomic_number = new int64_t[*numberOfParticlesPointer];
  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    if (map_species_z)
    {
      species_atomic_number[i] = z_map[particleSpeciesCodes[i]];
    }
    else { species_atomic_number[i] = particleSpeciesCodes[i]; }
  }

  // edge_distance_vec and edge_distances
  auto size_edges = graph.size();
  auto edge_index = std::vector<std::int64_t>(size_edges * 4);
  auto edge_distances = std::vector<double>(size_edges);
  auto edge_distances_vec = std::vector<double>(size_edges * 3);

  auto coords_2d = reinterpret_cast<double(*)[3]>(coordinates);
  auto _edge_distance_vec_ptr
      = reinterpret_cast<double(*)[3]>(edge_distances_vec.data());
  std::size_t index = 0;
  for (auto & edge_ : graph)
  {
    auto from_atom = edge_[0];
    auto to_atom = edge_[1];

    _edge_distance_vec_ptr[index][0]
        = coords_2d[to_atom][0] - coords_2d[from_atom][0];
    _edge_distance_vec_ptr[index][1]
        = coords_2d[to_atom][1] - coords_2d[from_atom][1];
    _edge_distance_vec_ptr[index][2]
        = coords_2d[to_atom][2] - coords_2d[from_atom][2];

    edge_distances[index] = std::sqrt(
        _edge_distance_vec_ptr[index][0] * _edge_distance_vec_ptr[index][0]
        + _edge_distance_vec_ptr[index][1] * _edge_distance_vec_ptr[index][1]
        + _edge_distance_vec_ptr[index][2] * _edge_distance_vec_ptr[index][2]);

    edge_index[index] = from_atom;
    edge_index[index + graph.size()] = to_atom;
    edge_index[index + 2 * graph.size()] = to_atom;
    edge_index[index + 3 * graph.size()] = from_atom;
    index++;
  }

  // 1. pos tensor
  int effectiveNumberOfParticlePointers
      = (*numberOfParticlesPointer == 1) ? 2 : *numberOfParticlesPointer;

  std::vector<std::int64_t> shape({effectiveNumberOfParticlePointers, 3});
  ml_model->SetInputNode(0, coordinates, shape, false, true);

  // Fix for isolated atoms. Append a dummy particle, not in graph
  shape.clear();
  shape = {effectiveNumberOfParticlePointers};

  // 2. batch tensor
  auto batch = std::vector<std::int64_t>(effectiveNumberOfParticlePointers, 0);
  ml_model->SetInputNode(1, batch.data(), shape, false, true);


  // 3. natoms
  auto natoms = std::vector<std::int64_t> {effectiveNumberOfParticlePointers};
  shape.clear();
  shape = {1};
  ml_model->SetInputNode(2, natoms.data(), shape, false, true);

  // 4. atomic number tensor
  shape.clear();
  shape = {effectiveNumberOfParticlePointers};
  ml_model->SetInputNode(3, species_atomic_number, shape, false, true);

  // 5. edge index tensor
  shape.clear();
  shape = {2, static_cast<std::int64_t>(graph.size())};
  ;
  ml_model->SetInputNode(4, edge_index.data(), shape, false, true);
  ;
  // 6. edge distance tensor
  shape.clear();
  shape = {static_cast<std::int64_t>(graph.size())};
  ml_model->SetInputNode(5, edge_distances.data(), shape, false, true);

  // 7. edge distance vector
  shape.clear();
  shape = {static_cast<std::int64_t>(graph.size()), 3};
  ml_model->SetInputNode(6, edge_distances_vec.data(), shape, false, true);
}

// --------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

void TorchExportModelDriverImplementation::readParametersFile(
    KIM::ModelDriverCreate * modelDriverCreate, int * ier)
{
  // Read parameter files from model driver
  // ---------------------------------------
  int num_param_files;
  std::string const *param_file_name, *tmp_file_name;
  std::string const * param_dir_name;

  modelDriverCreate->GetNumberOfParameterFiles(&num_param_files);

  for (int i = 0; i < num_param_files; i++)
  {
    modelDriverCreate->GetParameterFileBasename(i, &tmp_file_name);
    if (tmp_file_name->substr(tmp_file_name->size() - 4) == "yaml"
        || tmp_file_name->substr(tmp_file_name->size() - 3) == "yml")
    {
      param_file_name = tmp_file_name;
    }
    // else if (tmp_file_name->substr(tmp_file_name->size() - 2) == "so") //
    // currently loading .so from env variable
    // {
    //   model_file_name = tmp_file_name;
    // }
    else
    {
      LOG_ERROR("File extensions do not match; only expected .param or .pt");
      *ier = true;
      return;
    }
  }

  // Get param directory to load model and parameters from
  modelDriverCreate->GetParameterFileDirectoryName(&param_dir_name);

  std::string full_qualified_file_name
      = *param_dir_name + "/" + *param_file_name;
  // std::string full_qualified_model_name
  //     = *param_dir_name + "/" + *model_file_name;

  std::string placeholder_string;

  auto yaml_parser = YAMLReader();
  auto yaml_status = yaml_parser.load(full_qualified_file_name);
  if (yaml_status) { LOG_DEBUG("Loaded YAML file"); }
  else
  {
    LOG_ERROR("Could not load the YAML parameter file");
    *ier = true;
    return;
  }
  //
  // LOG_DEBUG("Successfully parsed parameter file");
  // if (*model_file_name != model_name)
  // {
  //   LOG_ERROR("Provided model file name different from present model file.");
  //   *ier = true;
  //   return;
  // }

  n_elements = std::stoi(yaml_parser.get("n_elements"));
  auto elements = yaml_parser.get("elements");

  std::stringstream element_stream(elements);

  auto element = std::string {""};
  for (int i = 0; i < n_elements; i++)
  {
    element.clear();
    if (!(element_stream >> element))
    {
      throw std::runtime_error("list of elements does not match n_elements");
    }
    else { elements_list.push_back(element); }
  }

  cutoff_distance = std::stod(yaml_parser.get("cutoff"));
  n_layers = std::stoi(yaml_parser.get("n_layers"));
  influence_distance = cutoff_distance * n_layers;
  auto device = yaml_parser.get("device");
  number_of_inputs = std::stoi(yaml_parser.get("number_of_inputs"));

  preprocessing = yaml_parser.get("preprocessing") == ""
                      ? "graph"
                      : yaml_parser.get("preprocessing");
  std::transform(preprocessing.begin(),
                 preprocessing.end(),
                 preprocessing.begin(),
                 ::tolower);  // ensuring consistent case

  // TEMP SOLUTION
  model_name
      = std::getenv(KIM_SO_ENV_VAR)
            ? std::getenv(KIM_SO_ENV_VAR)
            : "model.so";  // if no env var, load model.so from current dir


  // Load Torch Model
  // ----------------------------------------------------------------
  ml_model
      = MLModel::create(model_name.c_str(), device.c_str(), number_of_inputs);
  LOG_INFORMATION("Loaded Torch model and set to eval");
}

// --------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

void TorchExportModelDriverImplementation::unitConversion(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit,
    int * const ier)
{
  // DEFAULT units: A, eV, change if needed
  if (requestedLengthUnit != KIM::LENGTH_UNIT::A)
  {
    LOG_ERROR("Only Angstroms supported for length unit");
    *ier = true;
    return;
  }
  if (requestedEnergyUnit != KIM::ENERGY_UNIT::eV)
  {
    LOG_ERROR("Only eV supported for energy unit");
    *ier = true;
    return;
  }

  *ier = modelDriverCreate->SetUnits(KIM::LENGTH_UNIT::A,
                                     KIM::ENERGY_UNIT::eV,
                                     KIM::CHARGE_UNIT::unused,
                                     KIM::TEMPERATURE_UNIT::unused,
                                     requestedTimeUnit);
}

// --------------------------------------------------------------------------------
void TorchExportModelDriverImplementation::setSpecies(
    KIM::ModelDriverCreate * const modelDriverCreate, int * const ier)
{
  int code = 0;
  for (auto const & species : elements_list)
  {
    KIM::SpeciesName const specName1(species);

    //    std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator>
    //    modelSpeciesMap; std::vector<KIM::SpeciesName> speciesNameVector;
    //
    //    speciesNameVector.push_back(species);
    //    // check for new species
    //    std::map<KIM::SpeciesName const, int,
    //    KIM::SPECIES_NAME::Comparator>::const_iterator iIter =
    //    modelSpeciesMap.find(specName1);
    // all the above is to remove species duplicates
    *ier = modelDriverCreate->SetSpeciesCode(specName1, code);
    code += 1;
    if (*ier) return;
  }
}

// --------------------------------------------------------------------------------
void TorchExportModelDriverImplementation::registerFunctionPointers(
    KIM::ModelDriverCreate * const modelDriverCreate, int * const ier)
{
  // Use function pointer definitions to verify correct prototypes
  // TODO This doesn't look nice, implementation calling parent class
  // See if there is a workaround
  KIM::ModelDestroyFunction * destroy = TorchExportModelDriver::Destroy;
  KIM::ModelRefreshFunction * refresh = TorchExportModelDriver::Refresh;
  KIM::ModelComputeFunction * compute = TorchExportModelDriver::Compute;
  KIM::ModelComputeArgumentsCreateFunction * CACreate
      = TorchExportModelDriver::ComputeArgumentsCreate;
  KIM::ModelComputeArgumentsDestroyFunction * CADestroy
      = TorchExportModelDriver::ComputeArgumentsDestroy;

  *ier = modelDriverCreate->SetRoutinePointer(
             KIM::MODEL_ROUTINE_NAME::Destroy,
             KIM::LANGUAGE_NAME::cpp,
             true,
             reinterpret_cast<KIM::Function *>(destroy))
         || modelDriverCreate->SetRoutinePointer(
             KIM::MODEL_ROUTINE_NAME::Refresh,
             KIM::LANGUAGE_NAME::cpp,
             true,
             reinterpret_cast<KIM::Function *>(refresh))
         || modelDriverCreate->SetRoutinePointer(
             KIM::MODEL_ROUTINE_NAME::Compute,
             KIM::LANGUAGE_NAME::cpp,
             true,
             reinterpret_cast<KIM::Function *>(compute))
         || modelDriverCreate->SetRoutinePointer(
             KIM::MODEL_ROUTINE_NAME::ComputeArgumentsCreate,
             KIM::LANGUAGE_NAME::cpp,
             true,
             reinterpret_cast<KIM::Function *>(CACreate))
         || modelDriverCreate->SetRoutinePointer(
             KIM::MODEL_ROUTINE_NAME::ComputeArgumentsDestroy,
             KIM::LANGUAGE_NAME::cpp,
             true,
             reinterpret_cast<KIM::Function *>(CADestroy));
}

//******************************************************************************
int TorchExportModelDriverImplementation::ComputeArgumentsDestroy(
    KIM::ModelComputeArgumentsDestroy * const modelComputeArgumentsDestroy)
{
  // Nothing to do here?
  // As per my understanding, the ComputeArgumentsDestroy is used for
  // de-allocation and destruction of array compute arguments. We have no such
  // allocation
  TorchExportModelDriver * modelObject;  // To silence the compiler
  modelComputeArgumentsDestroy->GetModelBufferPointer(
      reinterpret_cast<void **>(&modelObject));
  return false;
}


//-------------------------------------------------------------------------------------
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

void TorchExportModelDriverImplementation::contributingAtomCounts(
    const KIM::ModelComputeArguments * modelComputeArguments)
{
  int * numberOfParticlesPointer;
  int * particleContributing;
  auto ier = modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
                 &numberOfParticlesPointer)
             || modelComputeArguments->GetArgumentPointer(
                 KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
                 &particleContributing);
  if (ier)
  {
    LOG_ERROR("Could not get number of particles @ contributingAtomCount");
    return;
  }
  n_contributing_atoms = 0;
  for (int i = 0; i < *numberOfParticlesPointer; i++)
  {
    if (particleContributing[i] == 1) { n_contributing_atoms += 1; }
  }
}


// *****************************************************************************
TorchExportModelDriverImplementation::~TorchExportModelDriverImplementation()
{
  delete[] graph_edge_indices;
  delete[] species_atomic_number;
  delete[] contraction_array;
}

// *****************************************************************************

int sym_to_z(std::string & sym)
{
  // TODO more idiomatic handling of species. Ask Ryan
  if (sym == "H") return 1;
  if (sym == "He") return 2;
  if (sym == "Li") return 3;
  if (sym == "Be") return 4;
  if (sym == "B") return 5;
  if (sym == "C") return 6;
  if (sym == "N") return 7;
  if (sym == "O") return 8;
  if (sym == "F") return 9;
  if (sym == "Ne") return 10;
  if (sym == "Na") return 11;
  if (sym == "Mg") return 12;
  if (sym == "Al") return 13;
  if (sym == "Si") return 14;
  if (sym == "P") return 15;
  if (sym == "S") return 16;
  if (sym == "Cl") return 17;
  if (sym == "A") return 18;
  if (sym == "K") return 19;
  if (sym == "Ca") return 20;
  if (sym == "Sc") return 21;
  if (sym == "Ti") return 22;
  if (sym == "V") return 23;
  if (sym == "Cr") return 24;
  if (sym == "Mn") return 25;
  if (sym == "Fe") return 26;
  if (sym == "Co") return 27;
  if (sym == "Ni") return 28;
  if (sym == "Cu") return 29;
  if (sym == "Zn") return 30;
  if (sym == "Ga") return 31;
  if (sym == "Ge") return 32;
  if (sym == "As") return 33;
  if (sym == "Se") return 34;
  if (sym == "Br") return 35;
  if (sym == "Kr") return 36;
  if (sym == "Rb") return 37;
  if (sym == "Sr") return 38;
  if (sym == "Y") return 39;
  if (sym == "Zr") return 40;
  if (sym == "Nb") return 41;
  if (sym == "Mo") return 42;
  if (sym == "Tc") return 43;
  if (sym == "Ru") return 44;
  if (sym == "Rh") return 45;
  if (sym == "Pd") return 46;
  if (sym == "Ag") return 47;
  if (sym == "Cd") return 48;
  if (sym == "In") return 49;
  if (sym == "Sn") return 50;
  if (sym == "Sb") return 51;
  if (sym == "Te") return 52;
  if (sym == "I") return 53;
  if (sym == "Xe") return 54;
  if (sym == "Cs") return 55;
  if (sym == "Ba") return 56;
  if (sym == "La") return 57;
  if (sym == "Ce") return 58;
  if (sym == "Pr") return 59;
  if (sym == "Nd") return 60;
  if (sym == "Pm") return 61;
  if (sym == "Sm") return 62;
  if (sym == "Eu") return 63;
  if (sym == "Gd") return 64;
  if (sym == "Tb") return 65;
  if (sym == "Dy") return 66;
  if (sym == "Ho") return 67;
  if (sym == "Er") return 68;
  if (sym == "Tm") return 69;
  if (sym == "Yb") return 70;
  if (sym == "Lu") return 71;
  if (sym == "Hf") return 72;
  if (sym == "Ta") return 73;
  if (sym == "W") return 74;
  if (sym == "Re") return 75;
  if (sym == "Os") return 76;
  if (sym == "Ir") return 77;
  if (sym == "Pt") return 78;
  if (sym == "Au") return 79;
  if (sym == "Hg") return 80;
  if (sym == "Ti") return 81;
  if (sym == "Pb") return 82;
  if (sym == "Bi") return 83;
  if (sym == "Po") return 84;
  if (sym == "At") return 85;
  if (sym == "Rn") return 86;
  if (sym == "Fr") return 87;
  if (sym == "Ra") return 88;
  if (sym == "Ac") return 89;
  if (sym == "Th") return 90;
  if (sym == "Pa") return 91;
  if (sym == "U") return 92;
  return -1;
}
