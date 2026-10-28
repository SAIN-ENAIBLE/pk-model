# For testing model.py in isolation 



###
### Model construction tests ###
###

def test_model_requires_central_params():
    """Model should only accept a CentralParams object."""
    with pytest.raises(TypeError)
    Model("not CentralParams")


def test_one_compartment_model():
    """No peripheral compartments should yield one total compartment."""
    params = CentralParams(
        central_volume = 5.0,
        clearance = 1.0,
    )

    model = Model(params)

    assert model.n_compartments() == 1

def test_three_compartment_model():
    """Two peripherals plus central should give three compartments."""
    peripheral_1 = PeripheralParams(
        volume = 10.0,
        flow = 2.0,
    )

    peripheral_2 = PeripheralParams(
        volume = 20.0,
        flow = 0.5,
    )

    params = CentralParams(
        central_volume = 5.0,
        clearance = 1.0,
        peripherals=(peripheral_1, peripheral_2),
    )

    model = Model(params)

    assert model.n_compartments() == 3