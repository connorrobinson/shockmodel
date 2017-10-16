import numpy as np
import matplotlib.pyplot as plt
import prepost
import os
from scipy import integrate
import time

'''

run_model.py

PURPOSE:
    Set up an individual post-shock + pre-shock model

AUTHOR:
    Connnor Robinson, Aug 31st, 2017
'''

M = 0.5
R = 2.0
Flog = 11
f = 0.01


jobnum = 1

#Define constants
mh = 1.67e-24 #g
Rsun = 6.96e10 #cm
Lsun = 3.839e33 #ergs

spectag = 'test'
cooltag = 'cooling'

specpath = '/Users/Connor/Desktop/Research/shock/code/cloudy_code/po_spectra/'


#Calculate mu based on solar abundances
#Fully Ionized
X = 0.70
Y = 0.28
Z = 0.02
mu = (2*X + (3/4)*Y + (1/2)*Z)**(-1)

model = prepost.postshock_structure(Flog, M, R, Ncells = 10000, grid2d = True, logT = True, ctfile = 'coolinggrid.txt')

npoints = 1000
spectra = prepost.make_postshockflux(npoints, model, cooltag, specpath, spectag,\
            ctfile = '/Users/Connor/Desktop/Research/shock/code/cloudy_code/coolinggrid.txt',\
            basepath = '/Users/Connor/Desktop/Research/shock/code/cloudy_code/models/')


nh = prepost.preshock_structure(Flog, M, R, Ri = 5)
pre_name = prepost.make_preshockmodel(nh, Flog, spectag, specpath, R, Rstop = 0.1)

#Run the cloudy model
# os.chdir(specpath)
# os.system('cloudy ' +spectag)

#Open up the pre-shock spectra:
data= np.genfromtxt(specpath+spectag+'.con', usecols = [0, 4, 5], skip_header = 1)

c = 3e14
nu = c/data[:,0]
postshock = spectra[:,1]/nu
preshock_out = data[:,1]/nu
preshock_in = data[:,2]/nu

prepost.write_output(nu, postshock, preshock_in, preshock_out, spectag, fill = 3, outpath = specpath)


Area = f * 4*np.pi * (R*Rsun)**2

Lacc = Area * 10**Flog / Lsun

Lpo = integrate.trapz(postshock, nu)/Lsun * Area
Lpre_in = integrate.trapz(preshock_in, nu)/Lsun * Area
Lpre_out = integrate.trapz(preshock_out, nu)/Lsun * Area




#Plot the post-shock structure
# fig, post_ax1 = plt.subplots()
# post_ax1.plot(model[0,:]/1e5, model[3,:]/1e5, color = 'r')
# post_ax1.set_xlabel(r'$z\times 10^5\;[cm]$')
# post_ax1.set_ylabel(r'$T\times 10^5\;[cm]$', color = 'r')
# post_ax1.set_ylim([0,9])
# post_ax1.set_xlim([-2, 0])
#
# post_ax2 = post_ax1.twinx()
# post_ax2.plot(model[0,:]/1e5, model[2,:]*mu*mh*1e10, color = 'b')
# post_ax2.set_ylabel(r'$\rho \times 10^{-10}\;[g cm^{-3}]$', color = 'b')
# post_ax2.set_xlim([-2, 0])
# post_ax2.set_ylim([.4, 3])
# plt.show()

#Plot ionization fraction and temperature vs height
# preshock = np.genfromtxt(specpath+spectag+'.ovr', skip_header = 1, usecols = [0,1,6,7,8])
# fig, ax1 = plt.subplots()
# ax1.semilogx(preshock[:,0], preshock[:,1]/1e4, color = 'b')
# ax1.set_xlabel(r'$log(z)\;[cm]$')
# ax1.set_ylabel(r'$T\times10^4\;[K]$', color = 'b')
#ax1.set_ylim([0.5, 2.2])

# ax2 = ax1.twinx()
# ax2.semilogx(preshock[:,0], preshock[:,3]/(preshock[:,2]+preshock[:,3]), color = 'r')
# ax2.set_ylabel(r'$\frac{n_{HII}}{n_{HI}+n_{HII}}$', color = 'r')
#ax2.tick_params('y', colors='r')

# fig.tight_layout()
# plt.show()








