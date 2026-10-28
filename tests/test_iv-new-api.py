import numpy as np
from scipy.linalg import expm

from pk_model import dose
from pk_model import clear
from pk_model import regime
from pk_model.model import Model
from pk_model.protocol import Protocol
from pk_model.compartments import Comp, Peripheral


def test_iv_central_compartment_analytical_solution():
    
    peripherals = ()
    
    comp = Comp(central_volume=15,
                    peripherals=peripherals)
    
    regime = regime.IV()
    
    dose = dose.infusion(start = 0,
                         end = 30,
                         rate = 15)
    clear = clear.linear(rate = 20)
    
    iv = Model(dose = dose,
               regime = regime,
               comp = comp,
               clear = clear)
    
    times = np.linspace(0, 10, 50)
    expected = 100.0 * np.exp(-times / 5.0)
    
    result = iv.solve(t_end=48)
    assert np.allclose(result.y[0], expected, rtol=1e-5, atol=1e-8)
    
    
def test_two_compartment_constant_dose_analytical_solution(): #TODO(3): Implement new API here.
    return
    
    central = {"qc": 100.0, "vc": 5.0, "cl": 1.0}

    compartments = {"compartment1": {"q1": 0.0, "v1": 10.0, "Q1": 2.0}}

    dose = 10.0

    times = np.linspace(0, 20, 200)

    model = Protocol(
        dose=dose,
        central=central,
        compartments=compartments,
        t_span=(0, 20),
        t_eval=times,
        type="IV",
        Ka=None,
    )

    # Linear system: dy/dt = A @ y + b
    A = np.array(
        [
            [-0.6, 0.2],
            [0.4, -0.2],
        ]
    )
    b = np.array([dose, 0.0])

    y0 = np.array([100.0, 0.0])

    augmented = np.zeros((3, 3))
    augmented[:2, :2] = A
    augmented[:2, 2] = b

    expected = np.array(
        [(expm(augmented * t) @ np.append(y0, 1.0))[:2] for t in times]
    ).T

    assert model.solution.success

    np.testing.assert_allclose(model.y, expected, rtol=1e-6, atol=1e-8)

def test_iv_three_compartment_infusion_analytical_solution():
    return
    
    peripherals = [Peripheral(10,5),
                   Peripheral(12,2)]
    
    comp = Comp(central_volume=15,
                        peripherals=peripherals)
        
    regime = regime.IV()
    
    dose = dose.infusion(start = 0,
                            end = 30,
                            rate = 15)
    clear = clear.linear(rate = 20)
    
    iv = Model(dose = dose,
                regime = regime,
                comp = comp,
                clear = clear)
    
    #TODO(3): Solve linear system for expected solution, then compare.