import shock
import EDGE as edge
import numpy as np
import matplotlib.pyplot as plt
import pdb
import corner
from astropy.io import ascii
from scipy import optimize
import time

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

#Set up target
targ  = 'gmaur'
wtarg = 'hbc427'
tags = ['HSTv1', 'HSTv2']

cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/'
modelpath = '/Users/Connor/Desktop/Research/shock/veilmodels/test/'

#If it's necessary to change these, you can, but the defaults should be ok for most purposes
modeltag = targ
plottarg = targ
figpath = modelpath
paramfile = targ+'_job_params.txt'

#Set the number of MCMC iterations + burnin
Nruns = 5000
burnin = 1000
nzeros = 4

#PROBABLY DO NOT NEED TO CHANGE ANYTHING BELOW HERE
#Define some constants
Ri = 5 #Stellar radii
G = 6.67e-8 #cm^3 g^-1 s^-2
Msun = 2e33 #g
Rsun = 6.957e10 #cm

#Dummy value of f for running code. If the code is run with f too large, it will crash.
#Rescaling is done inside code.
dummy_f = .01
ctts = edge.loadPickle(targ, picklepath = cttspath)

#Load in the parameter table and grab the unique veiling values
table = ascii.read(modelpath+targ+'_job_params.txt', data_start = 1, delimiter = ',')
veils = np.unique(np.array(table['VEILING']))

#Find models with matching values of veiling
#NOTE: This will only work where the stellar parameters do not change (i.e., only F changes).
veilmodels = np.array([np.where(np.array(table['VEILING']) == v)[0] for v in veils])
tagmodels =  np.array([np.where(np.array(table['datatag']) == "'"+t+"'")[0] for t in tags])

#Now loop over each data tag
for datatag in tags:
    
    #Prepare an array to accept results from each set of models
#    bigsample = []
    chi2 = []
    
    #Loop over each value of veiling to ensure that each model selected uses a consistant veiling value
    for i, veil in enumerate(veils):
        print(i)
        
        #Grab the wtts spectra for the first model with a given veiling value
        #Need to make it grab the right wtts file (since they are now scaled using the CTTS spectra)
        wtts_model = table['jobnum'][np.intersect1d(veilmodels[(veils == veil)][0], tagmodels[(tags == datatag)])[0]]
        wtts = edge.loadPickle(targ+'_'+wtarg + '_' + str(wtts_model).zfill(nzeros), picklepath = modelpath)
        
        #Get all the values for F for a given veiling value, and trim off the extra quotation marks
        F = [x[1:-1] for x in np.array(table['BIGF'][veilmodels[veils == veil]])[0]]
        jobs = np.array(table['jobnum'][veilmodels[veils == veil][0]])
        
        #Set up the MCMC chain
        sampler = shock.chisqr(ctts, wtts, F, jobs, targ, datatag, MCMC = True, Nruns = Nruns, modelpath = modelpath, nzeros = nzeros)
        samples = sampler.chain[:, burnin:, :].reshape((-1, len(F)))
        
        f = [np.median(samples[:,x]) for x in np.arange(len(F))]
        
        #Record the chi2 value for the best fit given this value of veiling
        chi2.append(shock.chisqr(ctts, wtts, F, jobs, targ, datatag, f = f, modelpath = modelpath, nzeros = nzeros))
        
        #Stack the results from the MCMC up. NOT SURE THIS WORKS AT THE MOMENT (WAS ORIGINALLY USING VSTACK)
#        if i == 0:
#            bigsample = np.zeros(np.shape(samples))
#        else:
#            bigsample = np.vstack([bigsample,samples])
    
    #Make computer yell at you when it finishes
    for x in np.arange(10):
        print('\a')
        time.sleep(.2)
    
    #Calculate the probability for each value of veiling
    prob = np.exp(-np.array(chi2)/2)/np.sum(np.exp(-np.array(chi2)/2))
    
    #Fit a gaussian to the probabilities
    p0 = [0.2, .1, .5]
    coeff, var_matrix = optimize.curve_fit(gauss, veils, prob, p0=p0)
    gveils = np.linspace(0,1,300)
    fit = gauss(gveils, coeff[0], coeff[1], coeff[2])
    
    #Plot up the probability plot for the veiling
    plt.scatter(veils, prob, color = 'k', marker = 'o', s = 40, label = 'Data')
    
    #Add shading to show 1, 2 and 3 stdevs
    plt.plot(gveils, fit, color = 'k', label = 'Gaussian fit', lw = 2, alpha = .5)
    colors = ['b', 'g', 'r']
    stdevs = [0,1,2]
    #Loop over each stdev
    for i, stdev in enumerate(stdevs):
        xlow = [np.argmin(np.abs(gveils - (coeff[1] + coeff[2]*stdev) )), np.argmin(np.abs(gveils - (coeff[1] + coeff[2]*(1+stdev)) ))]
        xhigh =  [np.argmin(np.abs(gveils - (coeff[1] - coeff[2]*stdev) )), np.argmin(np.abs(gveils - (coeff[1] - coeff[2]*(1+stdev)) ))]
        
        plt.fill_between(gveils[xlow[0]:xlow[1]+1], fit[xlow[0]:xlow[1]+1], color = colors[i], alpha = .2)
        plt.fill_between(gveils[xhigh[1]:xhigh[0]+1], fit[xhigh[1]:xhigh[0]+1], color  = colors[i], alpha = .2)
    
    #Add the best fit veiling value to the plot
    plt.text(0.4, 0.04, r'$r_v = ' + str(coeff[1])[0:4] + '\pm' + str(coeff[2])[0:4]+'$', fontsize = 20)
    
    #Finish off the plot (e.g. ranges, saving ect.)
    plt.ylabel(r'$P\,[r_v]$', fontsize = 17)
    plt.xlabel(r'$r_v$', fontsize = 17)
    plt.ylim([0, 0.1])
    plt.xlim([0, .6])
    
    plt.savefig(figpath+targ+'_'+datatag+'_veiling_prob.pdf')
    plt.show()
    
    #Now plot up the triangle plot for the MCMC fit for the best model
    #First get the best value of veiling in the models that have been run
    bestveil = np.argmin(np.abs(coeff[1] - veils))
    #Get the jobs associated with bestveil
    bestjobs = np.array(table['jobnum'][veilmodels[bestveil]])
    bestF = [x[1:-1] for x in np.array(table['BIGF'][veilmodels[bestveil]])]
    #Re-run the MCMC one last time for the best fit
    bestsampler = shock.chisqr(ctts, wtts, bestF, bestjobs, targ, datatag,\
        maskfile = '/Users/Connor/Desktop/Research/shock/code/mask.dat',\
        modelpath = modelpath,\
        part_interp = True, MCMC = True, Nruns = Nruns, nzeros = nzeros)
    bestsamples = bestsampler.chain[:, burnin:, :].reshape((-1, len(F)))
    bestf = [np.median(bestsamples[:,x]) for x in np.arange(len(F))]
    
    #Add one more 'sigma' contours to my plot
    levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.6, 0.5) ** 2)
    
    labels = [r'$' + str(x).split('E')[0] + r'\times 10^{'+str(x).split('E+')[1]+'}\,\,[\%]$' for x in bestF]
    corner.corner(bestsamples*100, color = 'g', labels = labels, \
        plot_datapoints = 0, \
        plot_density = False, \
        fill_contours = True, \
        levels = levels)
        
    plt.savefig(figpath+targ+'_'+datatag+'_corner.pdf')
    plt.show()
    
    #Calculate the mass accretion rate in useful units
    Mctts = table['MASS'][bestveil]
    Rctts = table['RADIO'][bestveil]
    vs = np.sqrt(2*G*(Mctts*Msun)/(Rctts*Rsun)) * np.sqrt(1-1/Ri)
    mdot = 8*np.pi*(Rctts*Rsun)**2/vs**2 * np.sum(np.array([float(x) for x in bestF]) * bestf)*(365*24*60*60/Msun)
    
    bestchi2 = shock.chisqr(ctts, wtts, bestF, bestjobs, targ, datatag, f = bestf,\
        maskfile = '/Users/Connor/Desktop/Research/shock/code/mask.dat',\
        modelpath = modelpath,\
        part_interp = True, MCMC = False, nzeros = nzeros)
    
    #Make the final shock plot
    shock.modelplot(bestF,bestjobs, bestf, targ, wtarg, datatag,\
        wttspath = modelpath,\
        cttspath = cttspath,\
        modelpath = modelpath,\
        plotpath = modelpath,\
        mask = False, maskfile = '/Users/Connor/Desktop/Research/shock/code/mask.dat',\
        plottarg = '', chi2 = str(bestchi2)[0:5], nzeros = nzeros, mdot = mdot, xlim = [2e3, 7e3], ylim = [1e-15, 1e-8],\
        photometry = True, spectrim = 0, smooth = 1, loc = 'best', errors = False)
    

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))