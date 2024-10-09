#!/bin/bash

set -e


wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "${HOME}/conda"
source "${HOME}/conda/etc/profile.d/conda.sh"
source "${HOME}/conda/etc/profile.d/mamba.sh"

conda init --all

conda env create -f ci/conda-envs/dev_env.yml -n skretrieval --yes

git submodule update --init
