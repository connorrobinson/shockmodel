import numpy as np
import itertools
from astropy.io import ascii
import shock

## Set up paths ect.
path = '/Users/Connor/Desktop/Research/shock/models/gmaur/veil0.2/' #Where to write the jobfiles +  the parameter list
samplepath = '/Users/Connor/Desktop/Research/shock/code/' #Location of the sample file. For now, keeping it the same location as this file
outpath = '/Users/Connor/Desktop/Research/shock/models/gmaur/veil0.2/'
NAME = 'gmaur'
jobnumstart = 8

#Info for scaling the WTTS photosphere
datatag = 'HSTv3'
wtarg = 'hbc427'
d_wtts = 140
d_ctts = 140
Rwtts = 1.9

DISTANCE = [d_ctts] #Distance in pc

#Parameters for the model
veiling = 0.2
MASS     = [1.1] #Stellar mass in solar units
RADIO    = [1.7] #Stellar radius in solar units
BIGF     = ['2E+10', '2.5E+10'] #Energy flux, probably want to write it in quotes
TSTAR     = [4350] # Effective temperature

#Should not need to edit anything under here, unless you are changing the location of the code
#Directories to where the code is located

#Scale the WTTS spectra using veiling + the distance to the object
shock.scale(NAME, wtarg, datatag, veiling, Rwtts, d_wtts, d_ctts, photometry = 0,\
     wttspath = '/Users/Connor/Desktop/Research/shock/data/wtts/',\
     cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/',\
     plotpath = '/Users/Connor/Desktop/Research/shock/plotting/scaled/',\
     outpath  = outpath,\
     wttstag = 'HST', clob = 1)

filewtts = NAME+'_'+wtarg+'veil'+str(veiling)+'.dat'

DIRPROG = '/project/bu-disks/shared/shockmodels/PROGRAMS'
DIRDAT = '/project/bu-disks/shared/shockmodels/DATAFILES'

#This filling factor is set low so that the code will run with large F. It is rescaled later in the code,
#so there is probably no need to change it here.
FILLING  = [0.01] # Filling factor

#Open up a file and print the parameter names
f = open(path+NAME+'-job_params.txt', 'w') 
f.writelines('jobnum, DISTANCE, MASS, RADIO, BIGF, TSTAR, FILLING, VEILING \n')#gravity, Av \n') 

#Write each iteration as a row in the table
for ind, values in enumerate(itertools.product(DISTANCE, MASS, RADIO, BIGF, TSTAR, FILLING)):#, gravity, Av)):
    f.writelines(str(ind+jobnumstart)+', '+ str(values)[1:-1]+', '+str(veiling)+ '\n')
f.close()

#Open up the table
names = ['jobnum', 'DISTANCE', 'MASS', 'RADIO', 'BIGF', 'TSTAR', 'FILLING', 'VEILING']
table = ascii.read(path+NAME+'-job_params.txt', data_start = 1, delimiter = ',') 

for i, row in enumerate(table[table.colnames[:-1]]):
    shock.create(path,row,names[:-1],NAME,filewtts,samplepath=samplepath)