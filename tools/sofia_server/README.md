# Sofia Server
Simple HTTP Server which wraps sofia calls.

Most of the paths are relative to a common datacube parent path.


## Conda Setup
  - Make sure Miniconda/Conda is installed
  - Use a conda shell and navigate to this folder
  - execute `conda env create -f conda_env.yml`
  - The environment `b3d_sofia_server` should be created
  - To update the environment run `conda env update --file conda_env.yml --prune` when the environment is active or `conda env update --name b3d_sofia_server --file conda_env.yml --prune` if the environment is not active.

## Run Server
Activate conda environment and start the server with:
```shell
python main.py common/datacube/parent/path [-se sofia_executable] [-sc sofia_config]
```

## Endpoints
### /start (POST)
Start a new sofia search with the given parameter.

_Request Body_:
```json
{
  "input.region": "x_min,x_max,y_min,y_max,z_min,z_max",
  "input.data": "relative/path/to/cube.fits",
  // optional sofia params
  "sofia_params": {
    // Almost all sofia params are directly forwarded to sofia.
    // output.directory must be a relative path and gets mapped to the common parent path e.G. my/custom/path gets translated to common/datacube/parent/path/my/custom/path
    // input.region and input.data is filtered out
  }
}
```
_Response Body (success, http: 200)_:
```json
{
  "message": "a message",
  "search_hash": "<hash_value>" // Hash over the input parameters to identify the search
}
```

_Response Body (error, http: 400 for wrong input or 503 if there is already an active sofia process)_:
```json
{
  "message": "Error message"
}
```

### /result (POST)
Fetch the results of a search with a given search_hash.

_Request Body_:
```json
{
  "search_hash": "<hash_value>"
}
```

_Response Body (success, http: 200)_:

Sofia process finished. Either with success or with error. See [error_code](https://gitlab.com/SoFiA-Admin/SoFiA-2/-/wikis/SoFiA-2-Error-Codes) for more details.

```json
{
  "message": "a message",
  "error_code": 0, // See sofia error codes. Everything expect 0 is an error.
  "output_dir": "<relative_output_dir>" // output directory relative to the common parent path
}
```

_Response Body (error, http: 400 for wrong input or 503 if there is an active sofia process)_:
```json
{
  "message": "Error message"
}
```
