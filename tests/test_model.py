import numpy as np

from pk_model import clear, dose, regime
from pk_model.compartments import Comp, Peripheral
from pk_model.model import Model


def make_model(peripherals=()):
    return Model(
        dose=dose.infusion(start=0, end=30, rate=15),
        regime=regime.IV(),
        comp=Comp(central_volume=15, peripherals=peripherals),
        clear=clear.linear(cl=20),
    )


def test_rhs_central_only():
    # c_c = 15/15 = 1  ->  dq_c = 15 - 20*1 = -5
    out = make_model().rhs(1.0, np.array([15.0]))
    np.testing.assert_allclose(out, [-5.0])


def test_rhs_one_peripheral():
    # c_c = 1, c_p = 0, exchange = 5*(1-0) = 5
    # dq_c = 15 - 20 - 5 = -10, dq_p = 5
    model = make_model((Peripheral(volume=10, transition_rate=5),))
    out = model.rhs(1.0, np.array([15.0, 0.0]))
    np.testing.assert_allclose(out, [-10.0, 5.0])
