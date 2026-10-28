from scipy.linalg import expm
import numpy as np
from pk_model.protocol import IV

def test_central_compartment_analytical_solution():
    central = {"qc": 100.0, "vc": 5.0, "cl": 1.0}
    compartments = {}

    times = np.linspace(0, 10, 50)
    model = IV(dose=0.0, central=central, compartments=compartments, t_span=(0, 10),t_eval=times)

    times = np.linspace(0, 10, 50)
    result = model.solver(t_span=(0, 10),t_eval=times)

    expected = 100.0 * np.exp(-times / 5.0)

    assert np.allclose(result.y[0], expected, rtol=1e-5, atol=1e-8)



def test_two_compartment_constant_dose_analytical_solution():
    central = {"qc": 100.0, "vc": 5.0, "cl": 1.0}

    compartments = {
        "compartment1": {"q1": 0.0, "v1": 10.0, "Q1": 2.0}
    }

    dose = 10.0

    times = np.linspace(0, 20, 200)

    model = IV(
        dose=dose,
        central=central,
        compartments=compartments,
        t_span=(0, 20),
        t_eval=times,
    )

    # Linear system: dy/dt = A @ y + b
    A = np.array([
        [-0.6, 0.2],
        [0.4, -0.2],
    ])
    b = np.array([dose, 0.0])

    y0 = np.array([100.0, 0.0])

    augmented = np.zeros((3, 3))
    augmented[:2, :2] = A
    augmented[:2, 2] = b

    expected = np.array([
        (expm(augmented * t) @ np.append(y0, 1.0))[:2]
        for t in times
    ]).T

    assert model.solution.success

    np.testing.assert_allclose(
        model.y, expected, rtol=1e-6, atol=1e-8
    )

