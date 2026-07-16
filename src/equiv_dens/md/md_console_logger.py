"""
MD Console Logger - SimulationHook for conventional MD-style console output.

Prints step, time, energies, temperature, dipole moments, and instantaneous
simulation speed (ns/day) at configurable intervals.
"""

import time
import torch

import equiv_dens.compat  # noqa: F401 - apply T_co patch before schnetpack import
from schnetpack.md.simulation_hooks.basic_hooks import SimulationHook
from schnetpack import units as spk_units

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schnetpack.md import Simulator


class MDConsoleLogger(SimulationHook):
    """
    SimulationHook that prints conventional MD-style progress to the console.

    Displays: Step, Time(ps), E_pot, E_kin, E_tot, T(K), dipole (μx μy μz |μ| in Debye),
    and instantaneous ns/day (rate over the last logging interval only).

    Args:
        every_n_steps: Print every N steps (default 100).
        time_step_fs: Timestep in femtoseconds (default 0.5).
        energy_unit: Energy display unit (default 'kcal/mol').
    """

    def __init__(
        self,
        every_n_steps: int = 100,
        time_step_fs: float = 0.5,
        energy_unit: str = "kcal/mol",
    ):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.time_step_fs = time_step_fs
        self.energy_unit = energy_unit
        # Conversion from internal (kJ/mol) to display unit for kinetic energy
        self._kinetic_to_display = spk_units.convert_units("kJ/mol", energy_unit)
        self._start_time: float | None = None
        self._last_log_time: float | None = None
        self._last_ns_per_day: float | None = None
        self._has_dipole: bool | None = None  # Set on first log

    def on_simulation_start(self, simulator: "Simulator") -> None:
        """Record start time and print header."""
        self._start_time = time.time()
        self._last_log_time = None

        header = (
            "Step    Time(ps)   E_pot     E_kin     E_tot     T(K)   "
            "μx(D)   μy(D)   μz(D)   |μ|(D)   ns/day"
        )
        print(header)

    def on_step_finalize(self, simulator: "Simulator") -> None:
        """Print progress line every every_n_steps."""
        if simulator.step % self.every_n_steps != 0:
            return

        system = simulator.system
        step = simulator.step

        # Energies: E_pot already in user units, E_kin in internal (kJ/mol)
        e_pot = system.energy.detach().cpu().squeeze().item()
        e_kin_internal = system.kinetic_energy.detach().cpu().squeeze().item()
        e_kin = e_kin_internal * self._kinetic_to_display
        e_tot = e_pot + e_kin

        # Temperature (Kelvin)
        t_k = system.temperature.detach().cpu().squeeze().item()

        # Time in ps
        time_ps = step * self.time_step_fs / 1000.0

        # Dipole: μx μy μz |μ| in Debye (or --- when not available)
        if "dipole_moment" in system.properties:
            if self._has_dipole is None:
                self._has_dipole = True
            d = system.properties["dipole_moment"].detach().cpu()
            # Shape: (n_replicas, n_mols*3) or (n_replicas, n_mols, 3)
            d_flat = d.view(-1, 3)
            mu = d_flat[0].numpy()
            mu_mag = (mu[0] ** 2 + mu[1] ** 2 + mu[2] ** 2) ** 0.5
            dipole_str = f"{mu[0]:7.2f} {mu[1]:7.2f} {mu[2]:7.2f} {mu_mag:6.2f}"
        else:
            if self._has_dipole is None:
                self._has_dipole = False
            dipole_str = "   ---     ---     ---     ---"

        # Instantaneous ns/day (last interval only)
        t_now = time.time()
        if self._last_log_time is not None:
            elapsed_s = t_now - self._last_log_time
            if elapsed_s > 0:
                ns_simulated = self.every_n_steps * self.time_step_fs / 1e6
                ns_per_day = ns_simulated / (elapsed_s / 86400.0)
                ns_str = f"{ns_per_day:6.1f}"
                self._last_ns_per_day = ns_per_day
            else:
                ns_str = "   ---"
        else:
            # First print: use cumulative as fallback
            if self._start_time is not None and (t_now - self._start_time) > 0:
                ns_simulated = step * self.time_step_fs / 1e6
                elapsed_s = t_now - self._start_time
                ns_per_day = ns_simulated / (elapsed_s / 86400.0)
                ns_str = f"{ns_per_day:6.1f}"
                self._last_ns_per_day = ns_per_day
            else:
                ns_str = "   ---"

        self._last_log_time = t_now

        line = (
            f"{step:6d}   {time_ps:7.2f}   {e_pot:7.2f}   {e_kin:7.2f}   {e_tot:7.2f}   "
            f"{t_k:6.1f}   {dipole_str}   {ns_str}"
        )
        print(line)

    def on_simulation_end(self, simulator: "Simulator") -> None:
        """Print brief summary."""
        if self._start_time is None:
            return
        elapsed_s = time.time() - self._start_time
        step = simulator.step
        ns_total = step * self.time_step_fs / 1e6
        ns_per_day_final = (
            self._last_ns_per_day
            if self._last_ns_per_day is not None
            else (ns_total / (elapsed_s / 86400.0) if elapsed_s > 0 else 0.0)
        )

        print()
        print(f"Simulation complete: {step} steps, {ns_total:.4f} ns, {elapsed_s/60:.1f} min")
        print(f"Final instantaneous speed: {ns_per_day_final:.1f} ns/day")
