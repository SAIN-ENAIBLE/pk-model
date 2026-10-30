import pytest

from pk_model import dose


@pytest.fixture
def infusion_dose():
    return dose.infusion(start=0,
                         end=30,
                         rate=15)

@pytest.mark.parametrize(
    argnames="t, expected_rate",
    argvalues=
    [
        pytest.param(0,15, id="start-inclusive"),
        pytest.param(29.999, 15, id="just-before-end"),
        pytest.param(30,0, id="end-exclusive"),
    ],
)
def test_infusion_range(infusion_dose, t, expected_rate):
    assert infusion_dose(t) == expected_rate

def test_negative_time(infusion_dose):    
    with pytest.raises(ValueError):
        infusion_dose(t=-2)

@pytest.mark.parametrize(
    argnames="time, expected_amount",
    argvalues=
    [
        pytest.param(0.0,0, id="start"),
        pytest.param(41.999,0.0, id="just-before"),
        pytest.param(42.0, 42.0, id="instant-bolus-time"),
        pytest.param(42.001,0.0, id="just-after")
    ]
    
)
def test_bolus_spike(time,expected_amount):
    
    instant_bolus = dose.bolus(amount=42,
                               time=42)
    assert instant_bolus(t=time) == pytest.approx(expected_amount)



            
