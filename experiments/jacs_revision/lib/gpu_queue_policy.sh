#!/usr/bin/env bash
# equiv_dens_ml GPU work stays off l40s_public, which is reserved for
# tps-cofolding production and the cofolding-boltzmann wrap allocation.
EQUIV_GPU_PARTITIONS=a100_chemistry,h200_public
EQUIV_GPU_QOS=gpu48
