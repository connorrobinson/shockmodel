import numpy as np
import itertools
from astropy.io import ascii
import shock

## Set up paths ect.
path = '/Users/Connor/Desktop/Research/shock/models/gmaur/cspect/' #Where to write the jobfiles +  the parameter list
samplepath = '' #Location of the sample file. For now, keeping it the same location as this file
NAME = 'cspect'
jobnumstart = 6

wttsfile = 'gmaur_hbc427.dat'

#Parameters for the model    
DISTANCE = [140] #Distance in pc
MASS     = [1.1] #Stellar mass in solar units
RADIO    = [1.7] #Stellar radius in solar units
BIGF     = ['1E+10', '1E+10.5', '1E+11', '1E+11.5', '1E+12'] #Energy flux, probably want to write it in quotes
Teff     = [4350] # Effective temperature


#Should not need to edit anything under here.

#This filling factor is set low so that the code will run with large F. It is rescaled later in the code,
#so there is probably no need to change it here.
FILLING  = [0.001] # Filling factor

#gravity  = [0.0] #log(g), but this value does not matter, recalculated by code anyway
#Av       = [0.0] #Extinction, but this value does not matter, never used by the code, at least in this iteration.

#Open up a file and print the parameter names
f = open(path+NAME+'-job_params.txt', 'w') 
f.writelines('jobnum, DISTANCE, MASS, RADIO, BIGF, FILLING, Teff \n')#gravity, Av \n') 

#Write each iteration as a row in the table
for ind, values in enumerate(itertools.product(DISTANCE, MASS, RADIO, BIGF, FILLING, Teff)):#, gravity, Av)):
    f.writelines(str(ind+jobnumstart)+', '+ str(values)[1:-1]+ '\n')
f.close()

#Open up the table
table = ascii.read(path+NAME+'-job_params.txt', data_start = 0, delimiter = ',') 

for i, row in enumerate(table):
    if i == 0:
        continue
    else:
        shock.create(path,row,table[0],NAME, wttsfile, samplepath=samplepath)