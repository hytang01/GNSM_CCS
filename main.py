import os
import numpy as np
import prep_Fine as prep_Fine
import run_ECL as run_ECL

def main():
    run_mode = 'sherlock' # 'local' or 'sherlock'
    
    path_curr = os.getcwd()
    
#     prep_Fine.set_ref_depth(path_curr)
    
    # run Coarse LGR
    run_ECL.run(path_curr, run_mode, 'Fine')
    
if __name__ == "__main__":
    main()