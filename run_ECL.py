import os
import subprocess
import shutil


def run(path_curr, run_mode, file_dir):

    eclipse_version = 'e300'
    if run_mode == 'sherlock':
        cmd = eclipse_version + '.exe'+' ' + 'CO2_ECLIPSE'
        os.system(cmd)
 

    