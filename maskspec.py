import EDGE as edge
import matplotlib.pyplot as plt
import numpy as np


'''

maskspec.py

Script to produce the wavelengths that I want to mask


'''

savepath = '/Users/Connor/Desktop/Research/shock/code/'

wtarg = 'hbc427'
targ  = 'gmaur'
datatag = 'HSTv2'

#Define the mask manually by looking at plot. Very similar to Ingleby 2015 excluded regions.

mask = [[4839,4895],\
        [4309,4383],\
        [4070,4120],\
        [3960,3981],\
        [3930,3946],\
        [3878,3908],\
        [3825,3859],\
        [3790,3812],\
        [3762,3730],\
        [2310,2360],\
        [2769,2845],\
        [2893,3000],\
        [3095,3160],\
        [   1,1700]]
    
#Set up paths
cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/'
wttspath = '/Users/Connor/Desktop/Research/shock/data/wtts/'

#Load the spectra
ctts = edge.loadPickle(targ,  picklepath = cttspath)
wtts = edge.loadPickle(wtarg, picklepath = wttspath)

#Plot the spectra
plt.figure(figsize = (18,5))
plt.plot(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4,ctts.spectra[datatag]['lFl'][np.argsort(ctts.spectra[datatag]['wl'])], color = 'g')
plt.plot(wtts.spectra['HST']['wl'][np.argsort(wtts.spectra['HST']['wl'])]*1e4, wtts.spectra['HST']['lFl'][np.argsort(wtts.spectra['HST']['wl'])], color = 'b', label = 'WTTS')

for region in mask:
    plt.fill_between(region, 0, 1e20, color = 'k', alpha =  .3)

plt.ylim([1e-15, 1e-8])
plt.yscale('log')

plt.show()

outfile = open(savepath+'mask.dat', 'w')

outfile.write('Value pairs of wavelengths in Angstroms that define masked areas\n')

for item in mask:
    outfile.write(str(item[0])+'   '+str(item[1])+'\n')

outfile.close()



