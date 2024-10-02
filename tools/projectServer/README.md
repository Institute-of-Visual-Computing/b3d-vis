# Sofia Server
Simple HTTP Server which wraps sofia calls.

All paramaters which contains a path or directory are relative to a parent path which is common to the server and the caller.


## Run Server
```
port, path_to_sofia_executable, common_root_path, 
```


## Endpoints
### /start (POST)
Start a new request with the given parameters.

_Request Body_:
```json
{
  // Unique string to identify the request
  "search_identifier" : "1234abcd",
  "sofia_config_file": "path to sofia config file relative to the common root path",
  // All sofia params are forwarded to sofia.
  "sofia_params": {
    // All path-like arguments like input.data or output.mask must be relative to the common root path for caller and server.
  }
}
```
_Response Body (success, http: 200)_:
```json
{
}
```

_Response Body (error, http: 400 for wrong input or 503 if there is already an active request)_:
```json
{
  "message": "Short description of the error"
}
```

### /result (POST)
Fetch the results of a request with a given search_identifier.

_Request Body_:
```json
{
  // Unique string to identify the request
  "search_identifier": "1234abcd"
}
```

_Response Body (success, http: 200)_:

Request finished. Either with success or with error. See [error_code](https://gitlab.com/SoFiA-Admin/SoFiA-2/-/wikis/SoFiA-2-Error-Codes) for more details.

```json
{
  "message": "a message",
  "result": {
    "sofiaResult": {
        "finished": true/false,
        "message": "SoFiA-2 message",
        "returnCode": 1
    }
  }
}
```

_Response Body (error, http: 400 for wrong input or 503 if the desired request is not finished)_:
```json
{
  "message": "Error message"
}
```

### /results (GET)

Returns all finished requests, if any. Empty json otherwise.

_Response Body (success, http: 200)_:
```json
{
    "<search_identifier_1>": {
        "sofiaResult": { ... }
    },
    "<search_identifier_2>": {
        "sofiaResult": { ... }
    },
    ...
}
```