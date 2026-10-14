"""The native suite has its own runtime; the Torch suite skips it explicitly."""
import pytest
import sys

jax=pytest.importorskip("jax", reason="Run native JAX contracts in the dedicated JAX runtime")


@pytest.fixture(autouse=True)
def release_compilation_memory():
    yield
    assert "torch" not in sys.modules, "Native qualification must not import the reference backend"
    # Tests compare unrelated static module trees; retain compilations within
    # each test (including geometry reuse), but bound memory across fixtures.
    jax.clear_caches()
