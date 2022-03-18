# Oneshell means all lines in a recipe run in the same shell
.ONESHELL:

# Specify conda environment name
CONDA_ENV=dt-jax

# Default device
DEVICE=gpu

# Need to specify bash in order for conda activate to work
SHELL=/bin/bash

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


env: sync-conda-env-name conda-env-update pip-compile pip-sync pre-commit echo-device

cpu-env: DEVICE=cpu
cpu-env: env

gpu-env: DEVICE=gpu
gpu-env: env

# Change the conda environment name of environment.yml
# in order to math the name with the Makefile's env name.
sync-conda-env-name:
	sed "1s/name: .*/name: ${CONDA_ENV}/" environment-${DEVICE}.yml > temp.yml && rm environment-${DEVICE}.yml && mv temp.yml environment-${DEVICE}.yml

# Create or update conda env
conda-env-update:
	conda env update -f environment-${DEVICE}.yml --prune

# Compile exact pip packages
pip-compile:
	$(CONDA_ACTIVATE) $(CONDA_ENV) && pip-compile -o requirements-${DEVICE}.txt -v \
	requirements/${DEVICE}.in requirements/prod.in requirements/dev.in

# Install pip packages
pip-sync:
	$(CONDA_ACTIVATE) $(CONDA_ENV) && pip-sync requirements-${DEVICE}.txt

# Install pre-commit
pre-commit:
	$(CONDA_ACTIVATE) $(CONDA_ENV) && pre-commit install

# Echo device
echo-device:
	echo "Successfully installed environment for the device: ${DEVICE}"

# Optional Jupyter nbextention installation. Run after env setup, if you want.
jupyter-ext-install:
	$(CONDA_ACTIVATE) $(CONDA_ENV) && jupyter contrib nbextension install
