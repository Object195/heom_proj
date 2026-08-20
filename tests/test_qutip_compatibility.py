"""Small integration checks for the QuTiP 5 benchmark code paths."""

import unittest

import numpy as np

try:
    import qutip
except ModuleNotFoundError:
    qutip = None


@unittest.skipIf(qutip is None, "QuTiP is an optional benchmark dependency")
class QuTiPCompatibilityTests(unittest.TestCase):
    def test_lindblad_benchmark_uses_modern_mesolve_api(self):
        from benchmark.benchmark_mlp import run_lindbladian

        trajectory = run_lindbladian(np.linspace(0.0, 0.01, 3))

        self.assertEqual(trajectory.shape, (3,))
        self.assertTrue(np.isfinite(trajectory).all())
        self.assertAlmostEqual(trajectory[0], 1.0)

    def test_pseudomode_benchmark_uses_modern_mesolve_api(self):
        from benchmark.benchmark_lindbladian import run_pseudomode_model

        trajectory = run_pseudomode_model(np.linspace(0.0, 0.01, 3))

        self.assertEqual(trajectory.shape, (3,))
        self.assertTrue(np.isfinite(trajectory).all())
        self.assertAlmostEqual(trajectory[0], 1.0)

    def test_heom_benchmark_uses_modern_solver_api(self):
        from benchmark.benchmark_lindbladian import run_qutip_heom

        trajectories = run_qutip_heom(
            t_eval=np.linspace(0.0, 0.01, 3),
            depths=(1,),
        )

        trajectory = trajectories[1]
        self.assertEqual(trajectory.shape, (3,))
        self.assertTrue(np.isfinite(trajectory).all())
        self.assertAlmostEqual(trajectory[0], 1.0)


if __name__ == "__main__":
    unittest.main()
