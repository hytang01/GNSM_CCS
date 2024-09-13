# import standard modules
import os

def main():
    e300_path = '/oak/stanford/schools/ees/smart_fields/ecl/macros/@e300'
    data_path = 'CO2_ECLIPSE' 
    cmd = "{} {} {} 0".format(e300_path, data_path, 1)
    print(cmd)
    
    if os.path.exists('CO2_ECLIPSE.UNRST'):
        print('pass this run')
    else:
        os.system(cmd)
    
main()
