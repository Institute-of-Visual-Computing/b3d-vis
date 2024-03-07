from flask import Flask, request, jsonify, send_file, make_response
import subprocess, os
from pathlib import Path

app = Flask(__name__)

sofia_exec_path = "/mnt/d/projects/b3d/SoFiA-2/sofia"
sofia_default_config_path = "./sofia_default_config.par"

sofia_processes = None

allowed_sofia_overwrites = ["input.region", "input.data"]
'''
    Create directory for File_Region_combo
    Set output.directory to this directory
    Start Process with sofia_default_config_path and overwrite_params
    If a new request comes in while running: return error code
    If same request comes in -> overwrite

'''

# Placeholder variables to store data
start_data = None
result_data = None
my_process = None

def start_sofia_process(overwrite_params):
    ppp = [sofia_exec_path, sofia_default_config_path]
    ppp.extend(overwrite_params)
    return subprocess.run(ppp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    

def check_process_finished(process):
    return process.returncode is not None  # Returns True if the process has finished


def convert_json_params_to_string(json_obj):
    result_strings = [f"{key}={value}" for key, value in json_obj.items()]
    return result_strings


def create_directory(directory_name):
    try:
        Path(directory_name).mkdir()
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")


@app.route('/start', methods=['POST'])
def start():
    global start_data, sofia_processes
    if not check_process_finished(sofia_processes):
        return make_response(jsonify({'message': 'Sofia already running'}), 503)
    
    sofia_processes = None

    start_data = request.json  # Store the JSON payload

    overwrite_params = {}

    if "input.data" not in start_data:
        return make_response(jsonify({'message': 'input.data required'}), 400)

    if Path(start_data['input.data']).exists() and Path(start_data['input.data']).is_file() and Path(start_data['input.data']).suffix == '.fits':
        overwrite_params['input.data'] = start_data['input.data']
    else:
        return make_response(jsonify({'message': f'input.data not valid at path ${start_data['input.data']}'}), 400)

    if "input.region" in start_data:
        input_region_coords = start_data['input.region'].split(',')
        if len(input_region_coords) != 6:
            return make_response(jsonify({'message': 'input.region is zero indexed and must be in the form x_min,x_max,y_min,y_max,z_min,z_max'}), 400)

    overwrite_params['input.region'] = start_data['input.region']

    # Region is x_min, x_max, y_min, y_max, z_min, z_max
    dir_name = start_data['input.region'].split(',')
    dir_name.append(Path(start_data['input.data']).stem)
    dir_name = '_'.join(dir_name)
    create_directory(dir_name)

    overwrite_params['output.directory'] = f"./{dir_name}"
    overwrite_params = convert_json_params_to_string(overwrite_params)
    print(overwrite_params)
    sofia_processes = start_sofia_process(overwrite_params)

    return jsonify({'message': 'Start data received successfully'})

@app.route('/result', methods=['POST'])
def result():
    global result_data
    result_data = request.json  # Store the JSON payload

    # Region is x_min, x_max, y_min, y_max, z_min, z_max
    dir_name = result_data['input.region'].split(',')
    dir_name.append(Path(result_data['input.data']).stem)
    dir_name = '_'.join(dir_name)


    # Process the result data and create a file
    # For demonstration, let's assume the result_data contains file content
    file_content = result_data.get('file_content', '')
    with open('result_file.txt', 'w') as f:
        f.write(file_content)
    # Return the file for download
    return send_file('result_file.txt', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
