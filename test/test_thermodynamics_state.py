import unittest

from aiphy.experiment.thermodynamics_state import ideal_gas_state_scan_config
from aiphy.main import Theorist


class ThermodynamicsStateExperimentTests(unittest.TestCase):
    def test_state_observations_are_not_misclassified_as_geometry(self):
        theorist = Theorist(num_threads=1)
        theorist.register_experiment(ideal_gas_state_scan_config())
        model = theorist.specific["ideal_gas_state_scan"]

        geometry = {item.name for item in model.fetch_geometry_variables}

        self.assertNotIn("pressure", geometry)
        self.assertNotIn("volume", geometry)
        self.assertNotIn("temperature", geometry)


if __name__ == "__main__":
    unittest.main()
