# Imports

import pk_model as model

# Tests

def test_import():
    """Verifies that package can be imported."""
    assert model is not None

def test_package_version():
    """Verifies that package declares its version."""
    assert hasattr(model, "__version__")
