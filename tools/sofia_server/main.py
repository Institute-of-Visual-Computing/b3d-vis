from flask import Flask, request, jsonify, send_file, make_response
import subprocess, os
from pathlib import Path
import hashlib
import threading
import argparse
import errno, sys

app = Flask(__name__)

sofia_exec_path = Path('sofia')
sofia_default_config_path = Path('./sofia_default_config.par')

# Every path is relative from here
data_path = Path("/mnt/d/data/work/b3d_data/datacubes")
rel_output_path = Path("sofia_server/output")

special_sofia_overwrites = ["input.region", "input.data"]

sofia_return_code_messages = {
    0: "The pipeline successfully completed without any error.",
    1: "An unclassified failure occurred.",
    2: "A NULL pointer was encountered.",
    3: "A memory allocation error occurred. This could indicate that the data cube is too large for the amount of memory available on the machine.",
    4: "An array index was found to be out of range.",
    5: "An error occurred while trying to read or write a file or check if a directory or file is accessible.",
    6: "The overflow of an integer value occurred.",
    7: "The pipeline had to be aborted due to invalid user input. This could, e.g., be due to an invalid parameter setting or the wrong input file being provided.",
    8: "No specific error occurred, but sources were not detected either."
}

# Map of searches
searches = {

}

# Current active search
active_search = None

search_management_lock = threading.Lock()

def convert_dict_params_to_string_array(json_obj):
    result_strings = [f"{key}={value}" for key, value in json_obj.items()]
    return result_strings

def create_directory(directory_name: Path):
    try:
        directory_name.mkdir()
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")

def process_active_search():
    global active_search
    if active_search is None:
        return
    if not active_search.isStarted():
        # start process
        return
    if active_search.isFinished():
        active_search = None

class SourceSearch:
    def __init__(self, relative_data_path: str, region: str, sofia_params: dict|None):
        self.relative_path = relative_data_path
        self.region = region
        self.process = None
        self.sofia_params = sofia_params
        self.hash = hashlib.md5(b"{self.relative_path}/{self.region}/{self.sofia_params}").hexdigest()
        self.output_dir = None
        self.overwrite_params = {}

    def isRunning(self) -> bool:
        if self.process is None:
            return False
        if self.process.returncode is None:
            return self.process.poll() is None
        return False
    
    def isStarted(self) -> bool:
        return self.process is not None
    
    def isFinished(self) -> bool:
        if self.isStarted():
            return not self.isRunning()
        return False

    def isValid(self) -> tuple[bool, str]:
        global data_path

        message = ''
        is_valid = True

        input_region_coords = self.region.split(',')
        if len(input_region_coords) != 6:
            if message:
                message += '\n'
            message += 'input.region is zero indexed and must be in the form x_min,x_max,y_min,y_max,z_min,z_max!'
            is_valid = False
        
        input_file_path = data_path / Path(self.relative_path)
        if not (input_file_path.exists() and input_file_path.is_file() and input_file_path.suffix == '.fits'):
            if message:
                message += '\n'
            message += f'input.data not valid at relative path {self.relative_path}!'
            is_valid = False
        return (is_valid, message)    
            
    def default_output_dir_path(self) -> Path:
        dir_name = self.region.split(',')
        dir_name.append(Path(self.relative_path).stem)
        dir_name = '_'.join(dir_name)
        return data_path / rel_output_path / Path(dir_name)

    def start(self) -> tuple[bool, str]:
        if self.isStarted():
            return (False, 'Search was already started')
        input_file_path = data_path / Path(self.relative_path)
        self.overwrite_params['input.data'] = input_file_path.as_posix()
        self.overwrite_params['input.region'] = self.region

        # Add other params
        if self.sofia_params is not None:
            for key in self.sofia_params:
                if key in special_sofia_overwrites:
                    self.sofia_params.pop(key)

        if self.sofia_params is not None and 'output.directory' in self.sofia_params:
            self.output_dir = data_path / Path(self.sofia_params['output.directory'])
            self.sofia_params.pop('output.directory')
        else:
            self.output_dir = self.default_output_dir_path()
        
        self.overwrite_params['output.directory'] = self.output_dir.as_posix()

        create_directory(Path(self.overwrite_params['output.directory']))

        process_args = [sofia_exec_path.as_posix(), sofia_default_config_path.as_posix()]
        process_args.extend(convert_dict_params_to_string_array(self.overwrite_params))
        if self.sofia_params is not None:
            process_args.extend(convert_dict_params_to_string_array(self.sofia_params))

        # TODO: Pipe stdout and err to file.
        self.process = subprocess.Popen(process_args)
        return (True, "")

    def wasSuccessful(self) -> tuple[bool, str, int]:
        if not self.isFinished():
            return (False, 'SoFiA-2 is running.', 0)
        was_sucessful = self.process.returncode == 0
        if self.process.returncode in sofia_return_code_messages:
            return (was_sucessful, sofia_return_code_messages[self.process.returncode], self.process.returncode)
        return (was_sucessful, 'Unknown error', -1)


@app.route('/start', methods=['POST'])
def start():
    global active_search, searches
    request_data = request.json  # Store the JSON payload
    with search_management_lock:
        process_active_search()

        # if a search is currently running -> return
        if active_search is not None:
            return make_response(jsonify({'message': 'Sofia already running.'}), 503)
        
        if 'input.data' not in request_data:
                return make_response(jsonify({'message': 'input.data required'}), 400)
        
        if 'input.region' not in request_data:
                return make_response(jsonify({'message': 'input.region required'}), 400)
        
        new_search = SourceSearch(request_data['input.data'], request_data['input.region'], None if 'sofia_params' not in request_data else request_data['sofia_params'])

        (search_is_valid, invalid_message) = new_search.isValid()
        if not search_is_valid:
            return make_response(jsonify({'message': invalid_message}), 400)

        message = 'Search Started.'
        if new_search.hash in searches:
            message = '\nSearch with given parameters was already processed. Overwriting!'
        
        print(f'Run Sofia with: {new_search.overwrite_params}')
        new_search.start()
        searches[new_search.hash] = new_search
        active_search = new_search

        return make_response(jsonify({'search_hash': new_search.hash, 'message': message}), 200)

@app.route('/result', methods=['POST'])
def result():
    global active_search, searches
    request_data = request.json  # Store the JSON payload
    with search_management_lock:
        process_active_search()

        # result_data should contain a "hash"
        if 'search_hash' not in request_data:
            return make_response(jsonify({'message': 'search_hash required'}), 400)

        search_hash = request_data['search_hash']
        if search_hash not in searches:
            return make_response(jsonify({'message': 'Given hash is not in processed searches'}), 400)

        selected_search = searches[search_hash]

        if not selected_search.isFinished():
            return make_response(jsonify({'message': 'Requested search is not finished.'}), 503)

        (search_was_success, message, error_code) = selected_search.wasSuccessful()
        relative_output_path = "" if selected_search.output_dir is None else selected_search.output_dir.relative_to(data_path).as_posix()
        return make_response(jsonify({'message': message, 'output_dir': relative_output_path, 'error_code': error_code}), 200)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog='Sofia Server',
                    description='Start SoFiA-2 with HTTP-Requests')
    parser.add_argument('parent_path', type=Path, help='Path to datacube storage path. All sofia path parameters are relative to this directory.') 
    parser.add_argument('-se', '--sofia_executable', default='sofia', type=Path, help='absolute path to the sofia executable.')
    parser.add_argument('-sc', '--sofia_config', type=Path, help='Path to the default sofia parameters used for the wrapper. Either absolute or relative to \'parent_path\'')
    args = parser.parse_args()
    
    data_path = args.parent_path

    if args.sofia_executable:
        if not args.sofia_executable.exists():
            print(f'Path to sofia executable does not exist: {args.sofia_executable}.')
            exit(errno.ENOENT)
        sofia_exec_path = args.sofia_executable
    
    if args.sofia_config:
        new_path = Path()
        if args.sofia_config.is_absolute():
            new_path = args.sofia_config.absolute()
        else:
            new_path = (data_path / args.sofia_config).absolute()
        if not new_path.exists() or not new_path.is_file():
            print(f'Path to sofia config file does not exist: {args.sofia_config}')
            exit(errno.ENOENT)
        
        sofia_default_config_path = new_path
    print('Running wrapper with:')
    print(f'parent path: {data_path}')
    print(f'sofia executable: {sofia_exec_path}')
    print(f'default config: {sofia_default_config_path}')

    app.run(debug=True)
