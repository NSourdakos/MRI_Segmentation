import sys
import logging
import optparse
import json
import os
import gc
import numpy as np
import json
import warnings



def set_system_settings():
    sys.setrecursionlimit(50000)
    logging.getLogger().setLevel(logging.INFO)


def get_command_line_arguments():
    parser = optparse.OptionParser()
    parser.set_defaults(config='NO_DS_ConfigMRI_v3_0_UNET_PlusPlus.json')
    parser.set_defaults(mode='training')
    parser.set_defaults(load_checkpoint=None)
    parser.set_defaults(condition_value=0)
    parser.set_defaults(batch_size=None)
    parser.set_defaults(one_shot=False)


    parser.add_option('--mode', dest='mode')
    parser.add_option('--print_model_summary', dest='print_model_summary')
    parser.add_option('--config', dest='config')
    parser.add_option('--load_checkpoint', dest='load_checkpoint')
    parser.add_option('--condition_value', dest='condition_value')
    parser.add_option('--batch_size', dest='batch_size')
    parser.add_option('--one_shot', dest='one_shot')
    parser.add_option('--noisy_input_path', dest='noisy_input_path')
    parser.add_option('--clean_input_path', dest='clean_input_path')
    parser.add_option('--target_field_length', dest='target_field_length')


    (options, args) = parser.parse_args()

    return options


def load_config(config_filepath):
    '''
    This is the function that loads the config .json file
    '''
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        logging.error('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)





def get_valid_output_folder_path(outputs_folder_path):
    j = 1
    while True:
        output_folder_name = 'samples_%d' % j
        output_folder_path = os.path.join(outputs_folder_path, output_folder_name)
        if not os.path.isdir(output_folder_path):
            os.mkdir(output_folder_path)
            break
        j += 1
    return output_folder_path






def pretty_json_dump(values, file_path=None):

    if file_path is None:
        print(json.dumps(values, sort_keys=True, indent=4, separators=(',', ': ')))
    else:
        json.dump(values, open(file_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))


def dir_contains_files(path):
  '''
  This function checks if the directory from which we load the checkpoint, for either training or testing,
  contains anyhtin
  '''
  for f in os.listdir(path):
        if not f.startswith('.'):
            return True
  return False