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

