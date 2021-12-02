#!/usr/bin/env bash
# This was just to try to get some "Universal 3D" parser set up so I could hopefully try
# to use that to extract the U3D format data from within invivoALatlas.pdf. I didn't get
# that working.

# TODO prevent user from running this script normally + prompt them to source it instead
# (so activating environment works)
# https://stackoverflow.com/questions/60115420

CONDA_ENV_NAME=al_atlas

# TODO delete / comment
conda remove -n "${CONDA_ENV_NAME}" --all
#


# TODO only do the stuff in the following block if environment doesn't already exist.
# otherwise, just activate it.

conda config --add channels conda-forge
conda create -n "${CONDA_ENV_NAME}" -y
# TODO what else do we need to install to get it to work?
# does vtk / the exporter itself also install u3d implicitly anyway?
# maybe look into what uses u3d and use one of those things?
# TODO probably move stuff like this to an environment.yml or something anyway...
conda install -n "${CONDA_ENV_NAME}" -y u3d


conda activate "${CONDA_ENV_NAME}"

