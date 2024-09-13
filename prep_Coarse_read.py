###########################################################################
#imports
###########################################################################

import numpy as np
import math
import re
from itertools import islice

###########################################################################
#module global variables
###########################################################################



###########################################################################
#functions:
###########################################################################

###########################################################################

def readDimsFromInput(fileName):
#author: Dylan M. Crain
#purpose: This function is merely meant to read from the ADGPRS input file for 
#         incompressible flow for an ensemble, the grid dimensions and total
#         grid counts in the fine model. 
#
#         Notes: ~This assumes uniform grid dimensions. 
#                ~The code can easily be improved. Do so at a later date.
#
#inputs:
#      -fileName -> name of Incomp. AD-GPRS input file
#
#outputs:
#       -grids -> number of gridblocks in the x, y, and z directions
#       -gridDims -> dimensions of gridblocks in x, y, and z: in meters
#
#       ~Both returned as lists

  associations = {'DXV':0, 'DYV':1, 'DZ':2, 'DZV':2}
  gridDims = [0, 0, 0]

  openFile = open(fileName)

  for line in openFile:
    temp = line.rstrip()
    if temp == 'DIMENS':
      line = next(openFile)
      line_split = line.rstrip().split(' ')
      grids = [int(line_split[0]), int(line_split[1]), int(line_split[2])]
    elif temp == 'DXV' or temp == 'DYV' or temp == 'DZ'or temp == 'DZV':     
        if temp in list(associations.keys()):
            next_line = next(openFile)
            next_line_split = next_line.rstrip().split(' ')
            value = next_line_split[0].rstrip().split('*')
          #The '[:-1]' is used to eliminate the stop for ADGPRS input - '\'
            gridDims[associations[temp]] = float(value[1])
            
  openFile.close()

  return grids, gridDims 

###########################################################################
def readTopFromInput(fileName):
#author: Dylan M. Crain
#purpose: This function is merely meant to read from the ADGPRS input file for 
#         incompressible flow for an ensemble, the grid dimensions and total
#         grid counts in the fine model. 
#
#         Notes: ~This assumes uniform grid dimensions. 
#                ~The code can easily be improved. Do so at a later date.
#
#inputs:
#      -fileName -> name of Incomp. AD-GPRS input file
#
#outputs:
#       -grids -> number of gridblocks in the x, y, and z directions
#       -gridDims -> dimensions of gridblocks in x, y, and z: in meters
#
#       ~Both returned as lists

  openFile = open(fileName)

  for line in openFile:
    temp = line.rstrip()
    if temp == 'TOPS':
      next_line = next(openFile)
      next_line_split = next_line.rstrip().split(' ')
      value = next_line_split[0].rstrip().split('*')
      top  = float(value[1])
            
  openFile.close()
  return top
###########################################################################
def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]
###########################################################################
def readBlockProps(fileName, grids, item):
    skip_line = 6 # this includes everything above the actual data
    
    # Initialize the time steps
    time_steps = []
    
    # Find data block and get the time steps
    fid=open(fileName,'r')
    for _ in range(skip_line):
        next(fid)
    for line in fid:
        line_bits = re.split(r'\s+', line)
        line_bits_clean = [i for i in line_bits if i]
        if line_bits_clean == []:
            break
        time_steps.append(float(line_bits_clean[0]))
    fid.close()
    
    # Get the number of time steps
    steps = len(time_steps)
    num_cells = grids[0]*grids[1]*grids[2]
    
    # Initialize the data structures for saturation and pressure
    #sats = np.zeros((*model_dims, steps))
    data_object = np.zeros((num_cells, steps)) 
    
    #i=1
    cell_count = 0
    # loop through file every 6+steps lines (chunks of data) at a time
    with open(fileName,'r') as fid:
        while True:
            next_n_lines = list(islice(fid,skip_line+steps))
            if not next_n_lines:
                break
            col_name=re.split(r'\s+',next_n_lines[2])
            col_name_clean = [i for i in col_name if i]
            # if col name contains BGSAT, get the index
            ind=[]
            if item in col_name_clean:
                ind=indices(col_name_clean, item)
                chunck_list = []
                for k in range(skip_line,steps+skip_line):
                    chunck_bits = re.split(r'\s+',next_n_lines[k])
                    chunch_bits_clean = [chunck for chunck in chunck_bits if chunck]
                    chunck_list.append(chunch_bits_clean)
                for n in range(len(ind)):
                    for j in range(steps):
                        index = ind[n]
                        data_object[cell_count][j]=chunck_list[j][index]
                    cell_count = cell_count + 1
    # reshape data into 4D array
    data_object_reshape = data_object.reshape((grids[0],grids[1],grids[2],steps), order='F') 
    
    # Considering only the last time step
    data_object_last_time=data_object_reshape[:,:,:,steps-1]
    data_object_last_time_reshape = data_object_last_time.reshape(grids[0]*grids[1]*grids[2], order='F')
    
    return data_object_last_time_reshape
###########################################################################

def readRockProps(fileName, numGrids):
#author: Dylan M. Crain
#purpose: This program is to read in basic data, such as porosity and perm
#         from a simple text file (one data point per line), with the exception
#         of the first and last lines, into an array. 
#
#inputs:
#      - fileName -> reads from either perm of poro data input for fine
#                 -> realization
#      -numGrids -> total number of fine grids, e.g. Nx * Ny * Nz
#                -> found from grids result from readDimsFromInput()
#
#outputs:
#       -vectorQuant -> numpy vector of rock property

  vectorQuant = np.zeros(numGrids)
  countLines = 0
  
  for line in open(fileName):
    if countLines > 0 and countLines <= numGrids :
      vectorQuant[countLines - 1] = float(line.rstrip())
    countLines += 1  
  return vectorQuant      
  
###########################################################################

def readWellsFromFine(fileName):
    numOfInj = 0
    nameOfInj = []
        
    injFlag = 'WCONINJE'
    
    f = open(fileName)    
    for line in f:
        lineSimp = line.rstrip()
        if lineSimp.startswith(injFlag):
            while lineSimp != '/': #loops until the end of the definition
                lineSimp = next(f).rstrip() 
                if lineSimp != '/' and not lineSimp.startswith('--') and len(lineSimp)!=0: #second check needed due to structure
                    if lineSimp.split()[0] not in nameOfInj:
                        numOfInj += 1
                        nameOfInj.append(lineSimp.split()[0])
    f.close()
    
    return nameOfInj

###########################################################################
# Amy
# Read the well type
def readWellTypeFromFine(fileName, nameOfWells):
    injType = []
    compdat_names = []
    x = []
    y = []
    z = []
    compdat = []
    z_endpoints = []
    
    wellInfoFlag = 'COMPDAT'
    f = open(fileName)
        
    for line in f:
        lineSimp = line.rstrip()
        if lineSimp.startswith(wellInfoFlag):
            while lineSimp != '/':  #loops until the end of the definition
                lineSimp = next(f).rstrip()
                if lineSimp != '/'and not lineSimp.startswith('--') and len(lineSimp)!=0:
                    compdat_names.append(lineSimp.split()[0])
                    compdat.append(lineSimp)
                    x.append(lineSimp.split()[1])
                    y.append(lineSimp.split()[2])
                    z.append([lineSimp.split()[3],lineSimp.split()[4]])
    
    if len(nameOfWells) != 0:
        for inj in nameOfWells:
            temp_i = []
            temp_j = []
            temp_k = []
            for i in range(len(compdat_names)):
                if compdat_names[i] == inj:
                    temp_i.append(x[i])
                    temp_j.append(y[i])
                    temp_k.append(z[i])
            if len(temp_i) == 1:
                injType.append('vertical_1') # vertical well type 1 -- defined with 1 line
                z_endpoints.append([int(temp_k[0][0]),int(temp_k[0][1])])
            else: 
                if (temp_i[0] == temp_i[-1] and temp_j[0] == temp_j[-1]):
                    injType.append('vertical_2') # vertical well type 2 -- defined with all blocks
                elif temp_i[0] != temp_i[-1] or temp_j[0] != temp_j[-1]:
                    if temp_k[0] == temp_k[-1]:
                        injType.append('horizontal')
                    else:
                        injType.append('deviated')
                else:
                    raise Exception('Check well type')
                z_endpoints.append([int(temp_k[0][0]),int(temp_k[-1][0])])
                
                                                                                        
    f.close()
    return injType, compdat, compdat_names, z_endpoints
    
###########################################################################
def readWellInfoFromFine(fileName, keyword):
    wellInfo = []
    
    wellInfoFlag = keyword
    f = open(fileName)
    
    for line in f:
        lineSimp = line.strip()
        if lineSimp.startswith(wellInfoFlag):
            while lineSimp != '/':  #loops until the end of the definition
                lineSimp = next(f).strip()
                if lineSimp != '/'and not lineSimp.startswith('--') and len(lineSimp)!=0:
                    wellInfo.append(lineSimp)
    return wellInfo
###########################################################################
def genNumOfBlockPerWell(nameOfWells, f_compdat_names, wellType, z_endpoints):
    injNumOfBlock = []
    
    if len(nameOfWells) != 0:
        for i in range(len(nameOfWells)):
            wellName = nameOfWells[i]
            endpoints = z_endpoints[i]
            
            if wellType[i] == 'vertical_1':
                injNumOfBlock.append(int(endpoints[1])-int(endpoints[0])+1)
                
            if wellType[i] == 'vertical_2' or wellType[i] == 'deviated' or wellType[i] == 'horizontal':
                injNumOfBlock.append(f_compdat_names.count(wellName))

    return injNumOfBlock
###########################################################################
# Amy
# Read the well coordinates
def readWellCoordFromFine(nameOfWells, compdat, wellType, wellNumOfBlock):
    injCoord = []
    
    if len(nameOfWells) != 0:
        for i in range(len(nameOfWells)):
            temp = []
            wName = nameOfWells[i]
            wType = wellType[i]
            wNumOfBlock = wellNumOfBlock[i]
            for j in range(len(compdat)):
                compName = compdat[j].split()[0]
                if wName in compName:
                    if wType == 'vertical_1':
                        start = int(compdat[j].split()[3])
                        for k in range(wNumOfBlock):
                            temp.append([int(compdat[j].split()[1]),int(compdat[j].split()[2]),start+k])
                    else:
                        temp.append([int(compdat[j].split()[1]),int(compdat[j].split()[2]),int(compdat[j].split()[3])])
            injCoord.append(temp)        

    return injCoord
###########################################################################
       