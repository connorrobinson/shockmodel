import shock
import EDGE as edge
import numpy as np
import matplotlib.pyplot as plt
import pdb
import corner
from astropy.io import ascii
from multiprocessing import Pool
from functools import partial

'''
shock_analysis.py

PURPOSE:
    This is an analysis tool that takes the input from many Calvet 98 shock models and finds the optimal value of veiling and the associated scale factors.
    It works by finding the best scale factors for each value of veiling using an MCMC appoach using the emcee package. The median model from the MCMC chains
    is then compared for each veiling, and fit with a Gaussian. The peak of this gaussian is the best fit value of the veiling.
    
INPUTS:
    This is a script, so there are no formal inputs, but you will need to change the name of the target, the name of the associated WTTS used to set the photosphere, and the
    observation tags, (set previously upon the creation of the EDGE data file).

OPTIONAL INPUTS:
    The number of MCMC runs, ect can be modified as well. More detailed descriptions are shown below

AUTHOR:
    Connor Robinson, June 2, 2017
'''
#Set up target
targ  = 'gmaur'
wtarg = 'recx1'
tags = ['HSTv2', 'HSTv3', 'HSTv4', 'HSTv5']

cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/'
modelpath = '/Users/Connor/Desktop/Research/shock/veilmodels/gmaur/'
maskfile = '/Users/Connor/Desktop/Research/shock/code/mask.dat'

#If it's necessary to change these, you can, but the defaults should be ok for most purposes
nzeros = 4 #Zero padding
Nruns = 5000  #MCMC iterations
burnin = 1000 #Burn in
ncores = 4 #Number of cores for multiprocessings
Ri = 5 #Inner disk edge

modeltag = targ
plottarg = targ
figpath = modelpath
paramfile = targ+'_job_params.txt'

## PROBABLY DO NOT NEED TO CHANGE ANYTHING BELOW HERE
dummy_f = .01 #Dummy value of f for running code. If the code is run with f too large, it will crash.

def solveVeil(veil, table=None, datatag=None, ctts=None, wtarg=None, burnin=None, modelpath=None, nzeros=None, Nruns=None):
    '''
    solveVeil()
    
    PURPOSE:
        Function that finds the correct models for a given veiling/datatag and then runs the MCMC grid
        It exists as a function in order to utilize multi-processing.
    
    INPUTS: NOTE: ALL OF THE INPUTS ARE NECESSARY.
        veil: [float] Value for the veiling
        table: [astropy.ascii table] Table generated from the parameter file containing all the info about the models
        datatag: [str] Tag associated with a given data set (e.g. 'HSTv1')
        ctts: [EDGE obs object] Observation object for the CTTS
        wtarg: [str] Weak T Tauri star associated with the models
        burnin: [int] Burn in for the MCMC
        nzeros: [int] Zero padding for the job files
        Nruns: [int] Number of MCMC interations
    
    RETURNS:
        (veil, f, chi2): [tuple] Contains the veiling, the median value of the samples 
                         of f for each F, and the chi2 value of the median
    
    AUTHOR:
        Connor Robinson, May 19th, 2017
    '''
    #Get the jobs for each value of veiling + datatag
    print(veil)
    vmodels =   np.array(table['jobnum'][table['VEILING'] == veil])
    tagmodels = np.array(table['jobnum'][table['datatag'] == "'"+datatag+"'"])
    jobs = np.intersect1d(vmodels, tagmodels)
    
    #Solve for the best fit, then compute the chi2 value
    samples = shock.MCMCsolve(table, ctts, jobs, burnin = burnin, modelpath = modelpath, nzeros = nzeros, Nruns = Nruns)#, Nthreads = 3)
    
    F = [np.array(table['BIGF'][table['jobnum'] == x])[0][1:-1] for x in jobs]
    f = [np.median(samples[:,x]) for x in np.arange(len(F))]
    return veil, f, shock.chisqr(ctts, F, jobs, targ, datatag, f = f, modelpath = modelpath, nzeros = nzeros)

def gauss(x, *p):
    '''
    gauss()
    
    PURPOSE:
        Returns a gaussian. Used for fitting veiling
    
    INPUTS:
        x: [float] Number/array for the x value
        p: Amplitude, offset and standard deviation of the gaussian
    
    RETURNS:
        An array containing a gaussian evaluated at each x
    
    '''
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))



#Load in the data and parameter table and grab the unique veiling values
ctts = edge.loadPickle(targ, picklepath = cttspath)
table = ascii.read(modelpath+paramfile, data_start = 1, delimiter = ',')
veils = np.unique(np.array(table['VEILING']))

#Find models with matching values of veiling
#NOTE: This will only work where the stellar parameters do not change (i.e., only F changes).
veilmodels = np.array(table['jobnum'][np.array([np.where(np.array(table['VEILING']) == v)[0] for v in veils])])

#Initialize a pool for multiprocessing
pool = Pool()

for datatag in tags:
    results = []
    
    ##Solve for the best veilings + filling parameters using multiprocessing
    #Create a new function in which all the parameters are fixed except the veiling
    partial_solveVeil = partial(solveVeil, \
       table = table, datatag = datatag, ctts = ctts, \
       wtarg = wtarg, burnin = burnin, modelpath = modelpath, \
      nzeros = nzeros, Nruns = Nruns)
    
    #Now run it on multiple processers!
    results = pool.map(partial_solveVeil, veils)
    
    #Parse results from the multiprocessing into something more useful
    resultveils = np.array([x[0] for x in results])
    f = np.array([x[1] for x in results])
    chi2 = np.array([x[2] for x in results])
    
    ## Calculate the best value of the veiling based on a chi2 minimization
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
    plt.xlim([0, 1.0])
    plt.savefig(figpath+targ+'_'+datatag+'_veiling_prob.pdf')
    plt.show()
    
    ## Make the corner plot for the best model
    #First get the job for the best fit for all veilings
    bestveil = np.argmin(np.abs(coeff[1] - veils))
    
    #Grab the best jobs for a given datatag
    vmodels = np.array(table['jobnum'][veilmodels[bestveil]])
    tagmodels = np.array(table['jobnum'][table['datatag'] == "'"+datatag+"'"])
    bestjobs = np.intersect1d(vmodels, tagmodels)
    bestF = np.array( [table['BIGF'][table['jobnum'] == x][0][1:-1] for x in bestjobs])
    
    #Run the MCMC for the best models again
    bestsamples = shock.MCMCsolve(table, ctts, bestjobs, burnin = burnin, modelpath = modelpath, nzeros = nzeros, Nruns = Nruns)
    bestf = [np.median(bestsamples[:,x]) for x in np.arange(len(bestF))]
    
    #Add one more 'sigma' contours to my plot
    levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.6, 0.5) ** 2)
    
    #Set up the labels for the axes
    labels = [r'$' + str(x).split('E')[0] + r'\times 10^{'+str(x).split('E+')[1]+'}\,\,[\%]$' for x in bestF]
    
    #Make the corner plot!
    corner.corner(bestsamples*100, color = 'g', labels = labels, \
        plot_datapoints = 0, \
        plot_density = False, \
        fill_contours = True, \
        levels = levels)
        
    plt.savefig(figpath+targ+'_'+datatag+'_corner.pdf')
    plt.show()
    
    #Calculate the mass accretion rate in useful units
    mdot = shock.mdot(table['MASS'][bestveil], table['RADIO'][bestveil], bestF, bestf, Ri = Ri)
    
    
    #Calculate the chi2 for the best model
    bestchi2 = shock.chisqr(ctts, bestF, bestjobs, targ, datatag, f = bestf,\
        maskfile = maskfile,\
        modelpath = modelpath,\
        part_interp = True, MCMC = False, nzeros = nzeros)
    
    #Make the final shock plot
    shock.modelplot(bestF, bestjobs, bestf, targ, wtarg, datatag,\
        wttspath = modelpath,\
        cttspath = cttspath,\
        modelpath = modelpath,\
        plotpath = modelpath,\
        mask = False, maskfile = maskfile,\
        plottarg = '', chi2 = str(bestchi2)[0:5], nzeros = nzeros, mdot = mdot, xlim = [2e3, 7e3], ylim = [1e-15, 1e-8],\
        photometry = True, spectrim = 0, smooth = 1, loc = 'best', errors = False)
