TorchExport Model Driver for KIM-API
====================================

This model drived provides interface between torch inductor exported models (last tested against torch == 2.4, exported 
using `torch._export.aot_inductor`) and KIM-API interface. Below is the step-by-step guide
to use your own models with it.

## 1. Install correct Torch env(conda)
You need torch 2.4.0, and other dependencies. Easiest is to use the provided environment
file.

```shell
conda env create -f environment.yml
```

## 2. Correct KIM env

Install the KIM-API

```
conda install kim-api==2.3.0 -c conda-forge

```

Install other python dependencies, `kimpy` is a hard dependency for ASE calculator.

```shell
pip install kim-edn==1.4.1 kim-property==2.6.4 kim-query==4.0.0 kimpy==2.1.1 
```

## 3. Env variables
For torch export to run models correctly you need properly source the env variables, namely
`LD_LIBRARY_PATH` and `INCLUDE`.
```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib
export INCLUDE=$INCLUDE:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/include
```

## 4. Install the model driver
Currently you need to provide the `CMAKE_PREFIX_PATH` explicitly. Will be fixed in future versions.

```shell
git clone https://github.com/ipcamit/TorchExportModelDriver
CMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib/python3.10/site-packages/torch/share/cmake kim-api-collections-management install user TorchExportModelDriver
```

You should see output as:
```shell
Found local item named: TorchExport__MD_000000000000_000.
In source directory: /path/to/TorchExportDriver.
   (If you are trying to install an item from openkim.org
    rerun this command from a different working directory,
    or rename the source directory mentioned above.)

[ 25%] Built target MLModel
[100%] Built target TorchExport__MD_000000000000_000
Install the project...
-- Install configuration: "Release"
-- Installing: /home/user/.kim-api/2.3.0+v2.3.0.GNU.GNU.GNU.2022-07-11-20-25-52/model-drivers-dir/TorchExport__MD_000000000000_000/libkim-api-model-driver.so
-- Set runtime path of "/home/user/.kim-api/2.3.0+v2.3.0.GNU.GNU.GNU.2022-07-11-20-25-52/model-drivers-dir/TorchExport__MD_000000000000_000/libkim-api-model-driver.so" to ""

Success!
```

## 4. Install the model
The repo contains, an example model,
```shell
cd TorchExportDriver
kim-api-collections-management install user example_model
```

The example model only has a simple yaml file:
```yaml
n_elements: 89
elements: H He Li ... Np Pu
cutoff: 6.0
n_layers: 1
device: cpu
number_of_inputs: 7
```

Where:

|keyword| Meaning|
|:------|:-------|
|`n_elements` | How many elements does the model support? |
|`elements` | List of element symbols |
|`cutoff` | cutoff used to create the radial graphs |
|`n_layers` | number of convolutions |
|`device` | `cpu` or `cuda`, depending on whether the model was exported for CPU or CUDA |
|`number_of_inputs` | Number of inputs the model expect |

So please adjust is as per the model you want to run.

The model name is defined in the `example_model/CMakeLists.txt` file as `eSCN1__MO_000000000000_000`

```cmake
cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})
find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)


kim_api_items_setup_before_project(ITEM_TYPE "portableModel")
project(eSCN1__MO_000000000000_000) # << NAME OF THE MODEL
kim_api_items_setup_after_project(ITEM_TYPE "portableModel")

add_kim_api_model_library(
  NAME            ${PROJECT_NAME}
  DRIVER_NAME     "TorchExport__MD_000000000000_000"
  PARAMETER_FILES "file.yaml"
  )

```

You can also alter that if you wish. But keep in mind that same changes need to be made in example scripts as well.

## 5. Run your model

The driver looks for a file `model.so` in the current working directory, you can override this behaviour by setting env
variable, `KIM_MODEL_SO_PATH`
```shell
export KIM_MODEL_SO_PATH=/path/to/my_model.so
```

That should be it. You can now run your model.

# ASE Example:

```python
import numpy as np
import ase
from ase import Atoms
from ase.calculators.kim.kim import KIM

model = KIM("eSCN1__MO_000000000000_000")

atoms = Atoms("CO2", positions=[[0.0, 0, 0], 
                                [1.2, 0, 0], 
                                [2.4, 0, 0]], 
                     pbc=[True, True, True], 
                     cell = np.eye(3)*3)
atoms.calc = model

print("PE from exported eSCN:\n", atoms.get_potential_energy())
print("Forces from exported eSCN:\n", atoms.get_forces())

```

# LAMMPS example:

```shell
# Initialize KIM Model
kim init eSCN1__MO_000000000000_000 metal

# Load data and define atom type
read_data test_si.data
kim interactions Si
mass 1 28.0855

neighbor 1.0 bin
neigh_modify every 1 delay 0 check no

# Create random velocities and fix thermostat
velocity all create 300.0 4928459 rot yes dist gaussian
fix 1 all nvt temp 300.0 300.0 $(100.0*dt)

timestep 0.001
thermo 1
run    10

```

