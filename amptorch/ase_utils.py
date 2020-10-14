from ase.calculators.calculator import Calculator
import numpy as np

class AMPtorch(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, trainer):
        Calculator.__init__(self)

        self.trainer = trainer

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        predictions = self.trainer.predict([atoms])
        cal_forces = predictions["forces"][0]
        cal_atoms_index = predictions['cal_atoms_index'][0]
        self.results["forces"] = [[0,0,0] for atom in atoms]
        for i, index in enumerate(cal_atoms_index):
            self.results["forces"][index] = list(cal_forces[i])
        self.results["forces"] = np.array(self.results["forces"])
        self.results["energy"] = predictions["energy"][0]
