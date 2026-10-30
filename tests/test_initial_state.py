"""Validate initial amounts and inputs through the supported simulation API."""

import numpy as np
import pytest
from scipy.linalg import expm

from pk_model.clear import Clear
from pk_model.dose import Dose
from pk_model.model import CentralParams, Model, PeripheralParams
from pk_model.simulation import Simulation
from pk_model.solve import solve_regimen

# ---------------------------------------------------------------------
# Initial amounts and dosing inputs
# Check IV/SC solutions against independent matrix-exponential references.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("route", ["IV", "SC"])
@pytest.mark.parametrize("callable_input", [False, True])
@pytest.mark.parametrize("has_peripheral", [False, True])
def test_initial_amounts_and_inputs_match_exact_solution(
    route, callable_input, has_peripheral
):
    peripherals = (PeripheralParams(10.0, 2.0),) if has_peripheral else ()
    model = Model(CentralParams(5.0, 1.0, peripherals))
    clear = Clear(1.0)
    initial_state = [50.0, 20.0] if has_peripheral else [50.0]
    if route == "SC":
        initial_state.append(0.0)
    regimen = [Dose.bolus(100.0, 2.0)] if route == "SC" and not callable_input else []
    dose_rate = (
        (lambda t: 2.0 * t) if callable_input else (100.0 if route == "IV" else 0.0)
    )
    Ka = 0.5 if route == "SC" else None
    times = np.array([2.0, 4.0, 8.0, 10.0])
    simulation = Simulation(model, regimen, route, clear, Ka=Ka)
    solution = simulation.solve(
        10.0,
        t_start=2.0,
        t_eval=times,
        dose_rate=dose_rate,
        initial_state=initial_state,
    )

    # Augment the distribution matrix with time and a constant to represent
    # both constant input rates and callable ramps exactly.
    size = len(initial_state)
    matrix = np.zeros((size + 2, size + 2))
    matrix[0, 0] = -0.2
    if has_peripheral:
        matrix[:2, :2] = [[-0.6, 0.2], [0.4, -0.2]]
    if route == "SC":
        matrix[0, size - 1] = 0.5
        matrix[size - 1, size - 1] = -0.5
    matrix[-2, -1] = 1.0
    initial = np.array([*initial_state, 2.0, 1.0])
    if callable_input:
        matrix[size - 1 if route == "SC" else 0, -2] = 2.0
    elif route == "SC":
        initial[size - 1] = 100.0
    else:
        matrix[0, -1] = 100.0
    expected = np.array([(expm(matrix * (t - 2.0)) @ initial)[:size] for t in times]).T
    np.testing.assert_array_equal(solution.t, times)
    np.testing.assert_allclose(solution.y, expected, rtol=1e-7, atol=1e-8)

    standalone = solve_regimen(
        model,
        regimen,
        clear,
        (2.0, 10.0),
        t_eval=times,
        protocol=route,
        Ka=Ka,
        dose_rate=dose_rate,
        initial_state=initial_state,
    )
    np.testing.assert_array_equal(standalone.t, times)
    np.testing.assert_allclose(standalone.y, expected, rtol=1e-7, atol=1e-8)
