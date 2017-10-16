import numpy as np
import itertools
from astropy.io import ascii
import EDGE as edge
import shock

## Set up paths ect.
path = '/Users/Connor/Desktop/Research/shock/models/cloudymodels/test/' #Where to write the jobfiles +  the parameter list
clusterpath = '/projectnb/bu-disks/connorr/shock/test/' #location of where the models will be placed on the cluster
samplepath = '/Users/Connor/Desktop/Research/shock/code/' #Location of the sample file. For now, keeping it the same location as this file
wttspath = '/Users/Connor/Desktop/Research/shock/data/wtts/' #Location of the dereddened WTTS pickle files
cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/' #Location of the dereddened CTTS pickle files
plotpath = '/Users/Connor/Desktop/Research/shock/plotting/scaled/' #Location you want plots sent (if plotting is turned on in the shock.scale call below)

NAME = 'gmaur'
jobnumstart = 1

#Info for scaling the WTTS photosphere
datatag = ['HSTv1']
wtarg = 'hbc427'
wttstag = 'HST'

#Parameters for the model
DISTANCE = [140] #Distance in pc
VEILING  = list(np.linspace(0.0,.2,3))
MASS     = [1.1] #Stellar mass in solar units
RADIO    = [1.7] #Stellar radius in solar units
BIGF     = ['1E+10', '3E+10', '1E+11', '3E+11', '1E+12'] #Energy flux, probably want to write it in quotes
TSTAR    = [4350] # Effective temperature

#Change the padding on the job numbers.
nzeros   = 4

#Should not need to edit anything under here, unless you are changing the location of the code
#Directories to where the code is located
DIRPROG  = '/project/bu-disks/shared/SHOCK/PROGRAMS'
DIRDAT   = '/project/bu-disks/shared/SHOCK/DATAFILES'

#Set up the paths/files that prepost needs to run. Unless you are changing location of the code/cloudy, or
#want to use different cooling/opacity files should not need to change.
BASEPATH = '/project/bu-disks/shared/SHOCK/PREPOST/models/'
CTFILE   = '/project/bu-disks/shared/SHOCK/PREPOST/models/coolinggrid.txt'
COOLTAG  = 'cooling'
CLOUDY   = '/projectnb/bu-disks/connorr/cloudy/c17.00/source/cloudy.exe'
OPCFILE  = '/project/bu-disks/shared/SHOCK/PREPOST/models/opacitygrid.txt'

#This filling factor is set low so that the code will run with large F. It is rescaled later in the code,
#so there is probably no need to change it here.
FILLING  = [0.01] # Filling factor

#Open up a file and print the parameter names
f = open(path+NAME+'_job_params.txt', 'w') 
f.writelines('jobnum, datatag, DISTANCE, MASS, RADIO, BIGF, TSTAR, FILLING, VEILING \n')

#Write each iteration as a row in the table
for ind, values in enumerate(itertools.product(datatag, DISTANCE, MASS, RADIO, BIGF, TSTAR, FILLING, VEILING)):
    f.writelines(str(ind+jobnumstart)+', '+ str(values)[1:-1]+ '\n')
f.close()

#Open up the table
names = ['jobnum', 'datatag', 'DISTANCE', 'MASS', 'RADIO', 'BIGF', 'TSTAR', 'FILLING', 'VEILING']
table = ascii.read(path+NAME+'_job_params.txt', data_start = 1, delimiter = ',') 

ctts = edge.loadPickle(NAME,  picklepath = cttspath)
wtts = edge.loadPickle(wtarg, picklepath = wttspath)

for i, row in enumerate(table[table.colnames]):
    
    #Scale the WTTS spectra using veiling + the distance to the object
    shock.scale(ctts, wtts, row['datatag'][1:-1], row['VEILING'], row['RADIO'], row['DISTANCE'], row['jobnum'], photometry = 0,\
         outpath  = path,\
         wttstag = wttstag, clob = 1, plot = 0, nzeros = nzeros)
    
    wttsfile = NAME+'_'+wtarg+'_'+str(row['jobnum']).zfill(nzeros)+'.dat'
    
    shock.create(path, row, names, NAME, wttsfile, samplepath = samplepath, nzeros = nzeros, \
    DIRPROG = DIRPROG, DIRDAT = DIRDAT, BASEPATH = BASEPATH, CTFILE = CTFILE, COOLTAG = COOLTAG, CLOUDY = CLOUDY, OPCFILE = OPCFILE)
    

shock.create_runall(jobnumstart, table['jobnum'][-1], clusterpath, samplepath = samplepath, nzeros = 4)


