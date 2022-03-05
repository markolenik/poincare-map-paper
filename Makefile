SHELL = /bin/sh

# Directories & files
project_name := poincare-map-paper
scripts_dir := scripts
data_dir := data
figures_dir := figures
ode_dir := odes

# * Targets

.PHONY: conda-install
conda-install:
	@conda env create -f environment.yml


.PHONY: conda-uninstall
conda-uninstall:
	@conda env remove --name $(project_name)


.PHONY: conda-activate
conda-activate:
	@conda activate $(project_name)


.PHONY: conda-deactivate
conda-deactivate:
	@conda deactivate


## Compute all stuffs
.PHONY: compute
compute:
	@python $(scripts_dir)/compute.py

## Recompute all stuffs
.PHONY: recompute
recompute:
	@python $(scripts_dir)/compute.py --recompute


## Delete all compiled Python files
.PHONY: clean-output
clean-output:
	@echo "Removing output data and figures."
	@rm $(bifdat)


.PHONY: clean-all
clean-all: clean-output
	@find . -type f -name "*.py[co]" -delete
	@find . -type f -name "*.dat" -delete
	@find . -type d -name "__pycache__" -delete



# TODO: Add linter and tests
# TODO: Add conda install env etc
