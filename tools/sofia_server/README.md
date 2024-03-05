### Conda Setup
  - Make sure Miniconda/Conda is installed
  - Use a conda shell and navigate to this folder
  - execute `conda env create -f conda_env.yml`
  - The environment `b3d_sofia_server` should be created
  - To update the environment run `conda env update --file conda_env.yml --prune` when the environment is active or `conda env update --name b3d_sofia_server --file conda_env.yml --prune` if the environment is not active.
