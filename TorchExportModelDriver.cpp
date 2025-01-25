#include "TorchExportModelDriver.hpp"
#include "TorchExportModelDriverImplementation.hpp"
//==============================================================================
//
// This is the standard interface to KIM Model Drivers
//
//==============================================================================

//******************************************************************************
extern "C" {
int model_driver_create(KIM::ModelDriverCreate * const modelDriverCreate,
                        KIM::LengthUnit const requestedLengthUnit,
                        KIM::EnergyUnit const requestedEnergyUnit,
                        KIM::ChargeUnit const requestedChargeUnit,
                        KIM::TemperatureUnit const requestedTemperatureUnit,
                        KIM::TimeUnit const requestedTimeUnit)
{
  int ier;
  // read input files, convert units if needed, compute
  // interpolation coefficients, set cutoff, and publish parameters
  auto modelObject = new TorchExportModelDriver(modelDriverCreate,
                                                requestedLengthUnit,
                                                requestedEnergyUnit,
                                                requestedChargeUnit,
                                                requestedTemperatureUnit,
                                                requestedTimeUnit,
                                                &ier);

  if (ier)
  {
    // constructor already reported the error
    delete modelObject;
    return ier;
  }

  // register pointer to TorchMLModelDriverImplementation object in KIM object
  modelDriverCreate->SetModelBufferPointer(static_cast<void *>(modelObject));

  // everything is good
  ier = false;
  return ier;
}
}  // extern "C"

//==============================================================================
//
// Implementation of TorchExportModelDriver public wrapper functions
//
//==============================================================================

// ****************************** ********* **********************************
TorchExportModelDriver::TorchExportModelDriver(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit,
    int * const ier)
{
  implementation_
      = new TorchExportModelDriverImplementation(modelDriverCreate,
                                                 requestedLengthUnit,
                                                 requestedEnergyUnit,
                                                 requestedChargeUnit,
                                                 requestedTemperatureUnit,
                                                 requestedTimeUnit,
                                                 ier);
}

// **************************************************************************
TorchExportModelDriver::~TorchExportModelDriver() { delete implementation_; }

//******************************************************************************
// static member function
int TorchExportModelDriver::Destroy(KIM::ModelDestroy * const modelDestroy)
{
  TorchExportModelDriver * modelObject;
  modelDestroy->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
  delete modelObject;
  return false;
}

//******************************************************************************
// static member function
int TorchExportModelDriver::Refresh(KIM::ModelRefresh * const modelRefresh)
{
  TorchExportModelDriver * modelObject;
  modelRefresh->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));

  return modelObject->implementation_->Refresh(modelRefresh);
}

//******************************************************************************
// static member function
int TorchExportModelDriver::Compute(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArguments const * const modelComputeArguments)
{
  TorchExportModelDriver * modelObject;
  modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
  return modelObject->implementation_->Compute(modelComputeArguments);
}

//******************************************************************************
// static member function
int TorchExportModelDriver::ComputeArgumentsCreate(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate)
{
  TorchExportModelDriver * modelObject;
  modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
  return modelObject->implementation_->ComputeArgumentsCreate(
      modelComputeArgumentsCreate);
}

//******************************************************************************
// static member function
int TorchExportModelDriver::ComputeArgumentsDestroy(
    KIM::ModelCompute const * modelCompute,
    KIM::ModelComputeArgumentsDestroy * const modelComputeArgumentsDestroy)
{
  TorchExportModelDriver * modelObject;
  modelCompute->GetModelBufferPointer(reinterpret_cast<void **>(&modelObject));
  return modelObject->implementation_->ComputeArgumentsDestroy(
      modelComputeArgumentsDestroy);
}