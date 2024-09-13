import os
import prep_PSS_helper as helper
import prep_Coarse_read as readf

def set_ref_depth(path_curr):
    wellFile = 'base.sched'
    fine_data = 'CO2_ECLIPSE.DATA'
    wellFile_2 = 'base2.sched'
    
    
    grids, gridDims = readf.readDimsFromInput(fine_data)
    top = readf.readTopFromInput(fine_data)
    topblock_center = top + gridDims[2]/2
    
    numOfInj, nameOfWells = helper.readWellsFromFine(wellFile)
    wellType, f_compdat, f_compdat_names, z_endpoints = readf.readWellTypeFromFine(wellFile, nameOfWells)
    wellNumOfBlock = readf.genNumOfBlockPerWell(nameOfWells, f_compdat_names, wellType, z_endpoints)
    wellCoord = readf.readWellCoordFromFine(nameOfWells, f_compdat, wellType, wellNumOfBlock)
    
    ref_depth = []
    for i in range(len(nameOfWells)):
        ref_depth.append(topblock_center + (wellCoord[i][0][2] - 1) * gridDims[2])
            
    f_welspecs = helper.readWellInfoFromFine(wellFile, 'WELSPECS')
    new_welspecs = []
    for n in range(len(f_welspecs)):
        w_spec = f_welspecs[n]
        w_name = w_spec.split()[0]
        w_i = w_spec.split()[2]
        w_j = w_spec.split()[3]
        w_reference_depth = str(ref_depth[n])
        
        
        new_welspecs.append(w_name+ ' '+ 'FIELD '+str(w_i)+' '+str(w_j)+' '+ w_reference_depth+ ' GAS /') 
    
    f= open(wellFile_2,"w+")
    f.close()
    
    # openFile = open(wellFile)
    identifier1 = 'WELSPECS'
    identifier2 = '/'
    print_flag = True
    f= open(wellFile_2,"a")
    
    with open(wellFile, 'r') as openFile:
        for line in openFile:
            temp = line.rstrip()
            if temp == identifier1:
                print_flag = False
            if temp == identifier2:
                print_flag= True
            if print_flag:
                f.write(temp+'\n')
            else:
                f.write('WELSPECS'+'\n')
                f.write('\n'.join(map(str, new_welspecs))+'\n')
                for i in range(numOfInj):
                    next(openFile)
            
    f.close()
    # openFile.close()
    
    os.remove(wellFile)
    os.rename(wellFile_2, wellFile)  