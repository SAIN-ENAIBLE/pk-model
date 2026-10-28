from pk_model.dose import Dose


def test_bolus_dose():
    dose = Dose.bolus(amount=100, t=10)

    assert dose.amount == 100
    assert dose.t == 10
