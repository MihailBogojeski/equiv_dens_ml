#!/usr/bin/env python3
"""Levels of theory shared by the ORCA (CPU) and PySCF/gpu4pyscf (GPU) label paths.

Both engines have to be told the same thing in two different dialects, and the
two sets of labels are only interchangeable if that translation is exact. Keeping
the pair in one place is what makes it checkable: `calibrate_theory.py` runs a
geometry through both dialects of the same entry and compares the densities.

`orca_reference_keywords` drops the RI approximations so the calibration can
separate a genuine disagreement between the codes from the RIJCOSX/RI-J error
that production runs deliberately accept.
"""

from __future__ import annotations

from dataclasses import dataclass

AUXBASIS = "augccpvqzjkfit"


@dataclass(frozen=True)
class TheoryLevel:
    key: str
    label: str
    orca_keywords: str
    orca_reference_keywords: str
    pyscf_xc: str
    pyscf_basis: str
    d4: bool
    nlc: bool
    auxbasis: str = AUXBASIS
    nlcgrids_level: int = 1
    #: Engines allowed to produce production labels at this level. A level is
    #: restricted to one engine when the two cannot be made to agree closely
    #: enough to share a split (see `engines` notes on each entry below).
    engines: tuple[str, ...] = ("orca", "pyscf")

    @property
    def dens_tag(self) -> str:
        """Filename fragment identifying basis + functional, e.g. ``def2tzvpd_wb97m-v``."""
        return f"{basis_tag(self.pyscf_basis)}_{self.pyscf_xc}"


def basis_tag(basis: str) -> str:
    """Filesystem-safe form of a basis name (``def2-tzvpd`` -> ``def2tzvpd``)."""
    return basis.replace("-", "").replace("_", "").replace("*", "s").lower()


# PBE0/aug-cc-pVDZ without D4 is the level the QM7-X campaign is already running;
# it stays the default so that campaign keeps reproducing byte-identical inputs.
_LEVELS: dict[str, TheoryLevel] = {
    "pbe0_avdz": TheoryLevel(
        key="pbe0_avdz",
        label="PBE0/aug-cc-pVDZ",
        orca_keywords="PBE0 aug-cc-pVDZ TightSCF EnGrad",
        orca_reference_keywords="PBE0 aug-cc-pVDZ TightSCF EnGrad",
        pyscf_xc="pbe0",
        pyscf_basis="augccpvdz",
        d4=False,
        nlc=False,
    ),
    # NORI matters: ORCA 6 switches RI-J on by default for pure functionals,
    # which shifts the density by 2.4e-4 relative to PySCF. With NORI+DEFGRID3
    # the two codes agree on the water dimer to 9.3e-6 in fit-coefficient L2 and
    # 7.1e-7 in the Coulomb metric, so ORCA can extend the PySCF frames already
    # on disk instead of starting a separate dataset.
    #
    # The residual energy difference, 3.3e-5 Ha, is larger than the density
    # difference can explain and is the D4 term: ORCA's built-in D4 and the
    # dftd4 Python package differ slightly. It is a per-frame additive shift
    # that leaves the density untouched.
    "pbe_d4_avdz": TheoryLevel(
        key="pbe_d4_avdz",
        label="PBE-D4/aug-cc-pVDZ",
        orca_keywords="PBE D4 aug-cc-pVDZ NORI DEFGRID3 TightSCF EnGrad",
        orca_reference_keywords="PBE D4 aug-cc-pVDZ NORI DEFGRID3 TightSCF EnGrad",
        pyscf_xc="pbe",
        pyscf_basis="augccpvdz",
        d4=True,
        nlc=False,
        engines=("orca", "pyscf"),
    ),
    "pbe0_d4_avdz": TheoryLevel(
        key="pbe0_d4_avdz",
        label="PBE0-D4/aug-cc-pVDZ",
        orca_keywords="PBE0 D4 aug-cc-pVDZ TightSCF EnGrad",
        orca_reference_keywords="PBE0 D4 aug-cc-pVDZ TightSCF EnGrad",
        pyscf_xc="pbe0",
        pyscf_basis="augccpvdz",
        d4=True,
        nlc=False,
    ),
    # Reviewer 3 named this level explicitly, and it is what OMol25/CSH uses.
    # wB97M-V carries VV10 nonlocal correlation, so no D4 on top.
    #
    # ORCA-only on purpose. RIJCOSX is the only approximation that reaches the
    # 20-24 water clusters, and PySCF has no COSX to match it with; RIJK would
    # be accurate enough but its analytic gradient aborts in ORCA 6.1.1. Since
    # this dataset is the size-scaling experiment, a size-correlated switch
    # between exact and approximate integrals would contaminate exactly the
    # trend being measured, so every frame here takes the same approximation
    # from one engine.
    #
    # Measured on the water dimer, RIJCOSX against exact integrals: 1.45e-4 in
    # fit-coefficient L2 but only 2.8e-5 in the Coulomb metric, 1.8e-4 Ha in
    # energy and 1.1e-4 Ha/bohr in forces. With RI switched off ORCA and PySCF
    # agree to 3.8e-7 in the Coulomb metric, so RIJCOSX is the whole of the
    # difference and the codes themselves are consistent. The 2.8e-5 density
    # floor sits about two orders of magnitude below the errors the model makes,
    # and being a fixed approximation it largely cancels in relative energies.
    "wb97mv_def2tzvpd": TheoryLevel(
        key="wb97mv_def2tzvpd",
        label="wB97M-V/def2-TZVPD",
        orca_keywords="wB97M-V def2-TZVPD def2/J RIJCOSX DEFGRID3 TightSCF EnGrad",
        orca_reference_keywords="wB97M-V def2-TZVPD NORI DEFGRID3 TightSCF EnGrad",
        pyscf_xc="wb97m-v",
        pyscf_basis="def2-tzvpd",
        d4=False,
        nlc=True,
        engines=("orca",),
    ),
}

DEFAULT_LEVEL = "pbe0_avdz"


def get_level(key: str) -> TheoryLevel:
    try:
        return _LEVELS[key]
    except KeyError:
        raise KeyError(f"unknown theory {key!r}; known: {', '.join(sorted(_LEVELS))}") from None


def level_keys() -> list[str]:
    return sorted(_LEVELS)
