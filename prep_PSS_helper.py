###########################################################################
#imports
###########################################################################

import numpy as np
import math
import re

###########################################################################
#module global variables
###########################################################################



###########################################################################
#functions:
###########################################################################
# Amy
# Read the number of injectors, the number of producers, and their names
# Here we assume no prior knowledge of well name.
# Well names and numbers are extracted from 'WCONINJE' and 'WCONPROD' keywords
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
    
    return numOfInj, nameOfInj

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
            if (temp_i[0] == temp_i[-1] and temp_j[0] == temp_j[-1]) or len(temp_i) == 1:
                injType.append('vertical')
            elif temp_i[0] != temp_i[-1] or temp_j[0] != temp_j[-1]:
                if temp_k[0] == temp_k[-1]:
                    injType.append('horizontal')
                else:
                    injType.append('deviated')
            else:
                raise Exception('Check well type')
                
                                                                          
    f.close()
    return injType, compdat
###########################################################################
# Amy  
# Read from fine base.sched any of the followings: 
    #'WELSPECS', 'WCONPROD', 'WCONINJE', 'WELLSTRE', 'COMPDAT (this guy is already taken care with the previous function, but can still be read with this function)'
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
def generateWelspecs(f_welspecs, upsc_fact):
    pps_welspecs = []
    for w_spec in f_welspecs:
        w_name = w_spec.split()[0]
        w_i = w_spec.split()[2]
        w_j = w_spec.split()[3]
        w_reference_depth = w_spec.split()[4]
        
        w_i_new = int(w_i) + upsc_fact[0] -1
        w_j_new = int(w_j) + upsc_fact[1] -1
        
        pps_welspecs.append(w_name+ ' '+ 'G1 '+str(w_i_new)+' '+str(w_j_new)+' '+ w_reference_depth+ ' WATER 1* 1* 1* NO /') 

    
    return pps_welspecs

 ###########################################################################
def generateCompdat(f_compdat, upsc_fact):
    pps_compdat = []
    for w_compdat in f_compdat:
        w_name = w_compdat.split()[0]
        w_i = w_compdat.split()[1]
        w_j = w_compdat.split()[2]
        w_remain = w_compdat.split()[3:]
        
        w_i_new = int(w_i) + upsc_fact[0] -1
        w_j_new = int(w_j) + upsc_fact[1] -1
        
        pps_compdat.append(w_name+ ' '+str(w_i_new)+' '+str(w_j_new)+' '+ ' '.join(w_remain)) 

    
    return pps_compdat
 ###########################################################################
def generateWconinje(f_wconinje, nameOfWells):
    pps_wconinje = []
    for w_name in nameOfWells:
        w_injRates = []
        for w_coninje in f_wconinje:
            if w_coninje.split()[0] == w_name:
                w_injRates.append(float(w_coninje.split()[4]))
        w_injRate_avg = sum(w_injRates)/len(w_injRates)
        
        pps_wconinje.append(w_name+' WATER OPEN RATE' + ' ' + str(w_injRate_avg) + ' 1* 4007 /')
        
    return pps_wconinje

    
    
    
    
    
    
    
    
    
    
