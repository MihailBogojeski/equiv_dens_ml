"""
Unit tests for MDConsoleLogger SimulationHook.
"""

import pytest
import torch

import equiv_dens.compat  # noqa: F401
from equiv_dens.md.md_console_logger import MDConsoleLogger


class MockSimulator:
    """Minimal mock of schnetpack Simulator for testing."""

    def __init__(self, step=0, has_dipole=True):
        self.step = step
        self.system = MockSystem(has_dipole=has_dipole)


class MockSystem:
    """Minimal mock of schnetpack System for testing."""

    def __init__(self, has_dipole=True):
        self.energy = torch.tensor([[[-42.5]]])  # kcal/mol
        self.kinetic_energy = torch.tensor([[[52.3]]])  # kJ/mol (internal)
        self.temperature = torch.tensor([[[298.0]]])
        self.properties = {}
        if has_dipole:
            self.properties["dipole_moment"] = torch.tensor([[[0.5, 0.8, 0.6]]])


def test_md_console_logger_import():
    """MDConsoleLogger can be imported."""
    assert MDConsoleLogger is not None


def test_md_console_logger_instantiation():
    """MDConsoleLogger instantiates with defaults."""
    logger = MDConsoleLogger(every_n_steps=100, time_step_fs=0.5, energy_unit="kcal/mol")
    assert logger.every_n_steps == 100
    assert logger.time_step_fs == 0.5
    assert logger.energy_unit == "kcal/mol"
    assert logger._kinetic_to_display > 0


def test_md_console_logger_on_simulation_start(capsys):
    """on_simulation_start prints header."""
    logger = MDConsoleLogger(every_n_steps=100, time_step_fs=0.5)
    sim = MockSimulator(step=0)
    logger.on_simulation_start(sim)
    captured = capsys.readouterr()
    assert "Step" in captured.out
    assert "Time(ps)" in captured.out
    assert "E_pot" in captured.out
    assert "ns/day" in captured.out


def test_md_console_logger_on_step_finalize_prints_at_interval(capsys):
    """on_step_finalize prints when step % every_n_steps == 0."""
    logger = MDConsoleLogger(every_n_steps=100, time_step_fs=0.5)
    logger._start_time = 0.0
    logger._last_log_time = None

    sim = MockSimulator(step=100)
    logger.on_step_finalize(sim)
    captured = capsys.readouterr()
    assert "100" in captured.out
    assert "0.05" in captured.out  # time_ps


def test_md_console_logger_on_step_finalize_skips_when_not_interval(capsys):
    """on_step_finalize does not print when step % every_n_steps != 0."""
    logger = MDConsoleLogger(every_n_steps=100, time_step_fs=0.5)
    logger._start_time = 0.0

    sim = MockSimulator(step=50)
    logger.on_step_finalize(sim)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_md_console_logger_dipole_absent(capsys):
    """When dipole not in properties, prints --- for dipole columns."""
    logger = MDConsoleLogger(every_n_steps=100, time_step_fs=0.5)
    logger._start_time = 0.0
    logger._last_log_time = None

    sim = MockSimulator(has_dipole=False)
    sim.step = 100
    logger.on_step_finalize(sim)
    captured = capsys.readouterr()
    assert "---" in captured.out


def test_md_console_logger_on_simulation_end(capsys):
    """on_simulation_end prints summary."""
    logger = MDConsoleLogger(every_n_steps=100, time_step_fs=0.5)
    logger._start_time = 0.0
    logger._last_ns_per_day = 12.5

    sim = MockSimulator(step=500)
    logger.on_simulation_end(sim)
    captured = capsys.readouterr()
    assert "Simulation complete" in captured.out
    assert "500" in captured.out
    assert "12.5" in captured.out
