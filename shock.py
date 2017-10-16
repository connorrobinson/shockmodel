import itertools
import EDGE as edge
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
import pdb
import matplotlib.patches as mpatches
import os
from scipy import interpolate
import emcee
import corner
import pandas as pd
from astropy.io import fits
from scipy import optimize

''''
shock.py

Package that contains most of the code required to work with the Calvet 1998 shock models


FUNCTIONS:
    
    scale: (Reworked, 4/11/17)
        Scale WTTS spectra based on veiling + distance + radius of the star
    
    create: (done)
        Creates the shock job file
    
    modelplot: (done)
        Makes plots of the models + data
    
    chisqr:
        Performs chi-squared minimization to get optimum model parameters
    
    
NOTE: Requires EDGE. You can get the latest version at https://github.com/cespaillat/EDGE
    
'''

def scale(ctts, wtts, datatag, veiling, Rctts, dctts, jobnum, photometry = 0,\
     plotpath = '/Users/Connor/Desktop/Research/shock/plotting/scaled/',\
     outpath  = '/Users/Connor/Desktop/Research/shock/data/wtts/scaled/',\
     wttstag = 'HST', clob = 0, verbose = 1, plot = True, nzeros = 3, wtts_scalephot = False,\
     ctts_scalephot = False):
    '''
     shock.scale
     
     PURPOSE:
        Scale WTTS spectra based on veiling + distance + radius of the star
    
    INPUTS:
         targ: [str] CTTS you are trying to model.
         wtarg: [str] WTTS you are trying to scale.
         datatag: [str] Tag for the CTTS spectra you are comparing against.
         veiling: [float] Veiling at V band. 
         dctts: [float] Distance to the ctts in pc 
         jobnum: [str/int] Job number associated with the output scaled photosphere
     
     OPTIONAL INPUTS:
         photometry: [boolean] If True, will include photometry in the output file. Only want this for plotting purposes
         plotpath: [str] path for plots to end up at
         outpath: [str] path for the scaled spectrum to go
         clob:[boolean] if true, will overwrite previous file
         verbose: [boolean] if true, code will yell things at you
         plot: [boolean] if true, will make a plot of the scaled photosphere
    
    '''
    targ = ctts.name
    wtarg = wtts.name
    
    #Define some constants
    pc = 3.09e16 #m
    Rsun = 6.96e8 #m
    
    #Fix the job number
    
    jobn = str(jobnum).zfill(nzeros)
    
    #Find the v band magnitude for the CTTS
    Vwl = 0.554 #V band wavelength in microns
    maxdiff = .005
    
    #Originally taking veiling from photometry. This is not correct, since I am now solving for veiling from the spectra
    #Select a small range of wavelength to median over
    close = np.abs(ctts.spectra[datatag]['wl'] - Vwl) < maxdiff
    # Now check if there are valid points for scaling
    if np.sum(close) == 0:
        raise ValueError('shock.scale ERROR: NO VALID CTTS SPECTRA FOUND FOR SCALING. WAVELENGTH MUST BE WITHIN 10nm OF 554nm')
    #Finally take the median flux value
    Vflux = np.median(ctts.spectra[datatag]['lFl'][close])
    
    #Repeat for the WTTS
    
    if wtts_scalephot:
        allwl = np.hstack([wtts.photometry[x]['wl'] for x in wtts.photometry.keys()])
        allflux = np.hstack([wtts.photometry[x]['lFl'] for x in wtts.photometry.keys()])
        wclose = np.abs(allwl - Vwl) < maxdiff
        if np.sum(wclose) == 0:
            raise ValueError('shock.scale ERROR: NO VALID WTTS PHOTOMETRY FOUND FOR SCALING. WAVELENGTH MUST BE WITHIN 10nm OF 554nm')
        wVflux = np.median(allflux[wclose])
        
    else:
        wclose = np.abs(wtts.spectra[wttstag]['wl'] - Vwl) < maxdiff
        if np.sum(wclose) == 0:
            raise ValueError('shock.scale ERROR: NO VALID WTTS SPECTRA FOUND FOR SCALING. WAVELENGTH MUST BE WITHIN 10nm OF 554nm')
        wVflux = np.median(wtts.spectra[wttstag]['lFl'][wclose])
    
    #Flatten the photometry into a single array
    #Start with the CTTS
    phot = np.array([])
    photwl = np.array([])
    for x in ctts.photometry.keys():
        phot = np.hstack([phot, ctts.photometry[x]['lFl']])
        photwl = np.hstack([photwl, ctts.photometry[x]['wl']])
    #Repeat for WTTS
    wphot = np.array([])
    wphotwl = np.array([])
    for x in wtts.photometry.keys():
        wphot = np.hstack([wphot, wtts.photometry[x]['lFl']])
        wphotwl = np.hstack([wphotwl, wtts.photometry[x]['wl']])
    
    #Now scale the spectra to the surface of the star
    #The code takes in the flux in units of erg s^(-1) cm^(-2) Ang ^(-1) (at the star)
    #Need to scale the observed flux of the WTTS to the surface of the CTTS
    factor = (Vflux/wVflux) * ((dctts * pc)/(Rctts * Rsun))**2 * (1/(1+veiling))
    
    #Scale the spectra/photometry + put in the right units
    photflux = (wphot/(wphotwl*1e4))*factor
    spectra = (wtts.spectra[wttstag]['lFl'] / (wtts.spectra[wttstag]['wl']*1e4)) * factor
    
    #Write everything to a file
    outfile = open(outpath+targ+'_'+wtarg+'_'+jobn+'.dat', 'w')
    for i, item in enumerate(spectra):
        outfile.write('       ' + str(1e4*wtts.spectra[wttstag]['wl'][i]).ljust(9,'0') +      '       '+str(spectra[i]).zfill(8)+'\n')
    #if the photometry flag is turned on, include the photometry in the outputted file. Should only be used for plotting purposes
    if photometry == True:
        for i, item in enumerate(photflux):
            outfile.write('       ' + str(1e4*wphotwl[i]).ljust(9,'0') +'       '+str(photometryflux[i]).zfill(8)+'\n') 
    outfile.close()
    
    #Make another pickle with the scaled spectra
    wtts_scaled = edge.TTS_Obs(targ+'_'+wtarg+'_'+jobn)
    wtts_scaled.add_spectra(wttstag, wtts.spectra[wttstag]['wl'], spectra)
    wtts_scaled.add_photometry('scaled', wphotwl, photflux, verbose = False)
    
    #Write a new pickle file
    wtts_scaled.SPPickle(outpath, clob = clob)
    
    #Plot up everything to make sure.
    if plot:
        plt.scatter(wphotwl, wphot * (Vflux/wVflux)*(1/(veiling+1)), color = 'b', zorder= 2 )
        plt.plot(ctts.spectra[datatag]['wl'], ctts.spectra[datatag]['lFl'], color = 'r')
        plt.plot(wtts_scaled.spectra[wttstag]['wl'], wtts_scaled.spectra[wttstag]['lFl'] * ((dctts * pc)/(Rctts * Rsun))**-2 * (wtts.spectra[wttstag]['wl']*1e4), color = 'b')
        plt.scatter(photwl, phot, color = 'r')
        plt.ylim([10**np.floor(np.log10(np.min(np.hstack([phot, wphot])))),  10**np.ceil(np.log10(np.max(np.hstack([phot, wphot]))))])
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
    
def create(path, row, names, NAME, wttsfile, samplepath = '', nzeros=3, \
    DIRPROG = '/project/bu-disks/shared/shockmodels/PROGRAMS',\
    DIRDAT = '/project/bu-disks/shared/shockmodels/DATAFILES',\
    outpath = '',\
    BASEPATH ='/project/bu-disks/shared/SHOCK/PREPOST/models/',\
    CTFILE ='/project/bu-disks/shared/SHOCK/PREPOST/models/coolinggrid.txt',\
    COOLTAG ='cooling',\
    CLOUDY ='/projectnb/bu-disks/connorr/cloudy/c17.00/source/cloudy.exe',\
    OPCFILE ='/project/bu-disks/shared/SHOCK/PREPOST/models/opacitygrid.txt'):
    
    '''
    shock.create
    
    PURPOSE:
        Creates the shock job file
    
    INPUTS:
        path: location of the job parameter list
        row: A row from the table containing all of the parameters
        names: The first row in the table containing all of the parameters names.
        NAME: name associated with the model
    
    
    OPTIONAL INPUTS:
        samplepath: [String] Path to the location of the sample. Default is in this directory.
        nzeros: [Int] Zero padding in the job number, default is 3
        DIRPORG/DIRDAT: [String] Paths to where the shockmodels themselves live
        outpath: [String] Path to where the files will be written. Default is the current directory.
        BASEPATH: [String] Path to the top level directory containing the cloudy models
        CTFILE: [String] File containing the cooling table
        COOLTAG: [String] Name associated with files in the cooling table
        CLOUDY: [String] Path + name of the cloudy executable file
        OPCFILE:[String] File containing the opacity information
    
    OUTPUTS:
        Batch file containing the necessary code to the run the model
    
    '''
    
    ## Write the batch file
    #-------------------------------
    #Read in template matlab script
    paramfile = open(samplepath+'shock_template', 'r')
    fulltext = paramfile.readlines()     # All text in a list of strings
    paramfile.close()
    
    #Transform the text into one long string
    text = ''.join(fulltext)
    
    
    #Replace the dummy parameter in brackets with the parameter from the table
    for i, param in enumerate(names):
        if param == 'jobnum':
            continue
        if param == 'datatag':
            continue
        else:
            start = text.find(param + "='")+len(param+"='")
            end = start + len(text[start:].split("'")[0])
            
            if param == 'BIGF':
                text = text[:start] + row[i][1:-1] + text[end:]
            else:
                text = text[:start] + str(row[i]) + text[end:]
        
    
    #Replace the WTTS file
    start = text.find('set filewtts=')+len('set filewtts=')
    end = start +len(text[start:].split('\n')[0])
    text = text[:start] + wttsfile + text[end:]
    
    #Set the name of the file
    start = text.find("NAME='")+len("NAME='")
    end = start + len(text[start:].split("'")[0])
    text = text[:start] + NAME+str(row[0]).zfill(nzeros) + text[end:]
    
    #Set the paths
    start = text.find('DIRPROG=')+len('DIRPROG=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start] + DIRPROG + text[end:]
    
    start = text.find('DIRDAT=')+len('DIRDAT=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start] + DIRDAT + text[end:]
    
    #Set up the cloudy stuff
    start = text.find('set BASEPATH=')+len('set BASEPATH=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start]+ "'" + BASEPATH + "'"+ text[end:]
    
    start = text.find('set CTFILE=')+len('set CTFILE=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start]+ "'" + CTFILE + "'" + text[end:]
    
    start = text.find('set COOLTAG=')+len('set COOLTAG=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start]+ "'" + COOLTAG + "'" + text[end:]
    
    start = text.find('set CLOUDY=')+len('set CLOUDY=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start] + "'" + CLOUDY + "'" + text[end:]
    
    start = text.find('set OPCFILE=')+len('set OPCFILE=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start] + "'" + OPCFILE + "'" + text[end:]
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the job file
    newjob = open(outpath+'job'+str(row[0]).zfill(nzeros), 'w')
    newjob.writelines(outtext)
    newjob.close()
    

def create_runall(jobstart, jobend, clusterpath, outpath = '', samplepath = '', nzeros = 3):
    '''
    shock.create_runall()
    
    INPUTS:
        jobstart: [int] First job file in grid
        jobsend: [int] Last job file in grid
    
    OPTIONAL INPUTS:
        samplepath: Path to where the runall_template file is located. Default is the current directory.
    
    
    '''
    #Now write the runall script
    runallfile = open(samplepath+'runall_template', 'r')
    fulltext = runallfile.readlines()     # All text in a list of strings
    runallfile.close()
    
    #Turn it into one large string
    text = ''.join(fulltext)
    
    #Replace the path
    start = text.find('cd ')+len('cd ')
    end = start +len(text[start:].split('\n')[0])
    text = text[:start] + clusterpath + text[end:]
    
    #Replace the jobstart
    start = text.find('#qsub -t ')+len('#qsub -t ')
    end = start +len(text[start:].split('-')[0])
    text = text[:start] + str(int(jobstart)) + text[end:]
    
    #Replace the job end
    start = text.find('#qsub -t '+str(int(jobstart))+'-')+len('#qsub -t '+str(int(jobstart))+'-')
    end = start +len(text[start:].split(' runall.csh')[0])
    text = text[:start] + str(int(jobend)) + text[end:]
    
    #Replace nzeros
    start = text.find('job%0')+len('job%0')
    end = start +len(text[start:].split('d" $SGE_TASK_ID')[0])
    text = text[:start] + str(int(nzeros)) + text[end:]
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the runall file
    newrunall = open(outpath+'runall.csh', 'w')
    newrunall.writelines(outtext)
    newrunall.close()
    


def modelplot(F,jobnums, f, ctts, wtts, datatag,\
    modelpath ='/Users/Connor/Desktop/Research/shock/models/',\
    plotpath = '/Users/Connor/Desktop/Research/shock/plotting/modelplots/',\
    mask = False, maskfile = '/Users/Connor/Desktop/Research/shock/code/mask.dat',\
    plottarg = '', chi2 = -1, nzeros = 3, mdot = -1, xlim = [1e3, 2e4], ylim = [1e-14, 1e-8],\
    photometry = True, spectrim = 0, smooth = 1, loc = 'best', errors = False):
    '''
    shock.modelplot
    
    PURPOSE:
        Makes plots of the models + data
    
        Currently can handle 5 models, but more could be added (simply by adding more plotting colors)
    
    INPUTS:
        F: [list of strings] Energy flux. Should be an array in the form: ['1E+11','1E+12']
        jobnums: [list of ints] Job numbers associated with each F (in order)
        ctts:[EDGE observation object] Observation object for the CTTS
        wtts:[EDGE observation object] Observation object for the WTTS
        f: [list of floats] Filling factor. Should be an array in the form: [0.02,  0.001]
        datatag: [str] Tag associated with the spectrum for the ctts, e.g. 'HSTv1'
    
    OPTIONAL INPUTS:
        modelpath: [String] path to models
        plotpath: [String] path for created plot
        mask: [Boolean] If True, will overplot the mask in grey
        maskfile: [String] Path+name of the mask file, default is '/Users/Connor/Desktop/Research/shock/code/mask.dat'
        plottarg: [str] Name of the ctts that will be used for plotting  (e.g. plottarg = 'GM Auriga' while targ = 'gmaur')
        chi2: [Float] If set to anything other than -1 will print chi**2 on the plot
        photometry: [Boolean] If True, photometry will be included on plot
        spectrim: [list of ints] Pairs of wavelengths to trim the observed between. E.g., [[2100, 2300], [4500,4600]] in angstroms
        smooth: [Int] Window for smoothing. Automatically set to 1 (no smoothing) (Uses the pandas rolling_mean function)
        loc: ['String'] Sets the location of your legend
        fontsize:[int]:
        
        
    OUTPUTS:
        Shows + makes a plot of the model
    
    '''
    
    targ  = ctts.name
    wtarg = wtts.name
    
    #Fix the modelpath to include the target name if the default path is used
    if modelpath == '/Users/Connor/Desktop/Research/shock/models/':
        modelpath = modelpath+targ+'/'
    
    if len(F[0]) == 1:
        modelname = ['fort40.'+targ+str(jobnums).zfill(nzeros)]
    else:
        modelname = ['fort40.'+targ+str(job).zfill(nzeros) for job in jobnums]
        
    #Sum up the components
    wl, Fall, Fhp, Fpre, Fphot = modelsum(targ, f, F, jobnums, nzeros = nzeros, modelpath = modelpath)
    
    #Plot the data
    if spectrim == 0:
        plt.plot(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4, pd.rolling_mean(ctts.spectra[datatag]['lFl'][np.argsort(ctts.spectra[datatag]['wl'])],smooth), color = 'g')
    else:
        trimmed = np.array([], dtype = int)
        for region in spectrim:
            trimmed = np.hstack([trimmed, np.intersect1d(np.where(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4 > region[0]), np.where(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4 < region[1]))])
        spectrum_trimmed = ctts.spectra[datatag]['lFl'][np.argsort(ctts.spectra[datatag]['wl'])]
        
        #Mask data outside the range of the model
        spectrum_trimmed[trimmed] = np.nan
        plt.plot(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4, pd.rolling_mean(spectrum_trimmed, smooth), color = 'g')
    
    #Add the photometry to the plot
    if photometry == True:
        phot = np.hstack([ctts.photometry[x]['lFl'] for x in ctts.photometry.keys()])
        photwl = np.hstack([ctts.photometry[x]['wl'] for x in ctts.photometry.keys()])
        plt.scatter(photwl*1e4, phot, marker = 'o', s = 30, color = 'g')
    
    #Add Errors to the plot THIS CURRENTLY TAKES FOREVER TO PLOT!!!!
    if errors == True:
        plt.errorbar(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4, \
        ctts.spectra[datatag]['lFl'][np.argsort(ctts.spectra[datatag]['wl'])], \
        yerr = ctts.spectra[datatag]['err'][np.argsort(ctts.spectra[datatag]['wl'])], color = 'g', alpha = .5)
        #plt.fill_between(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4, \
        #(ctts.spectra[datatag]['err'] + ctts.spectra[datatag]['lFl'])[np.argsort(ctts.spectra[datatag]['wl'])], \
        #(ctts.spectra[datatag]['err'] - ctts.spectra[datatag]['lFl'])[np.argsort(ctts.spectra[datatag]['wl'])], color = 'g', alpha = .1)
        
    #Begin plotting the model
    #Defile colors to be used for the shock. If you want more than 5 components will need to add more colors
    colors = ['b', 'r', 'g', 'm', 'c', 'purple']
    
    legendhandles = []
    
    if len(modelname) == 1:
        plt.plot(wl[np.argsort(wl)], pd.rolling_mean(Fhp[np.argsort(wl)] * f,smooth), color = colors[0], alpha = .5, linestyle = '--', lw = 2)
        plt.plot(wl[np.argsort(wl)], pd.rolling_mean(Fpre[np.argsort(wl)] * f,smooth), color = colors[0], alpha = .5, linestyle = ':', lw = 2)
        legendhandles.append(mpatches.Patch(color=colors[0], label=r'$F = $'+F+r', $f = $'+str(np.round(f,decimals = 4))))
    
        plt.plot(wl[np.argsort(wl)], pd.rolling_mean(Fall[np.argsort(wl)],smooth), color = 'k', label = 'Total Flux')
        plt.plot(wl[np.argsort(wl)], pd.rolling_mean(Fphot[np.argsort(wl)] * (1-f),smooth), color = 'k', alpha = .5, label = 'Photosphere')
    
    else:
        for i, model in enumerate(modelname):
            plt.plot(wl[i][np.argsort(wl[i])], pd.rolling_mean(Fhp[i][np.argsort(wl[i])] * f[i],smooth), color = colors[i], alpha = .5, linestyle = '--', lw = 2)
            plt.plot(wl[i][np.argsort(wl[i])], pd.rolling_mean(Fpre[i][np.argsort(wl[i])] * f[i],smooth), color = colors[i], alpha = .5, linestyle = ':', lw = 2)
            legendhandles.append(mpatches.Patch(color=colors[i], label=r'$F = $'+F[i]+r', $f = $'+str(np.round(f[i],decimals = 4))))
            
        plt.plot(wl[0][np.argsort(wl[0])], pd.rolling_mean(Fall[np.argsort(wl[0])],smooth), color = 'k', label = 'Total Flux')
        plt.plot(wl[0][np.argsort(wl[0])], pd.rolling_mean(Fphot[0][np.argsort(wl[0])] * (1-f[i]), smooth), color = 'k', alpha = .5, label = 'Photosphere')
    
    if plottarg == '':
        plottarg = datatag
    
    FhpArtist = plt.Line2D((0,1),(0,0), color='k', linestyle='--', label = 'Heated Photosphere', lw = 2)
    FpreArtist = plt.Line2D((0,1),(0,0), color='k', linestyle=':', label = 'Preshock Region', lw = 2)
    FallArtist = plt.Line2D((0,1),(0,0), color='k', label = 'Total Flux')
    PhotArtist = plt.Line2D((0,1),(0,0), color='k', label = 'Photosphere', alpha = .5)
    DataArtist = plt.Line2D((0,1),(0,0), color='g', label = plottarg)
    
    #Overplot the mask if the flag is on
    if mask == True:
        maskraw = np.genfromtxt(maskfile, skip_header= 1)
        for region in maskraw:
            plt.fill_between(region, 0, 1e20, color = 'k', alpha =  .3)
    
    legendhandles.append(FhpArtist)
    legendhandles.append(FpreArtist)
    legendhandles.append(FallArtist)
    legendhandles.append(PhotArtist)
    legendhandles.append(DataArtist)
    
    #Add chi to the legend
    if chi2 != -1:
        chiArtist = plt.Line2D((0,1),(0,0), color='w', label = r'$\chi^2 = $'+str(chi2))
        legendhandles.append(chiArtist)
        
    #Add mdot to the legend
    if mdot != -1:
        mdotArtist = plt.Line2D((0,1),(0,0), color='w', label = r'$\dot{M} = '+str(np.round(mdot/10**(np.floor(np.log10(mdot))), decimals = 2)) +r'\times 10^{'+str(int(np.floor(np.log10(mdot))))+r'}\,M_{sun}/yr$') 
        legendhandles.append(mdotArtist)
        
    plt.xlabel(r'Wavelength $\rm[\AA]$')
    plt.ylabel(r'$\rm \lambda F_{\lambda}\,[erg \cdot s^{-1} \cdot cm^{-2}]$')  
    
    plt.yscale('log')
    
    plt.legend(loc = loc, handles = legendhandles, fontsize = 12)
    plt.ylim(ylim)
    plt.xlim(xlim)
    
    if plottarg == '':
        plottarg = targ
    
    plt.title(plottarg)
    plt.savefig(plotpath+targ+'_'+datatag+'.pdf')
    
    plt.show()
    
    
    
def chisqr(ctts, F, jobnums, datatag, f = None,
    maskfile = '/Users/Connor/Desktop/Research/shock/code/mask.dat',\
    modelpath ='/Users/Connor/Desktop/Research/shock/models/',\
    part_interp = True, MCMC = False, Nruns = 2500, nzeros = 3, Nthreads = 1):
    
    '''
    shock.chisqr
    
    PURPOSE:
        Calculates chi-squared for a given model
        
    INPUTS:
        
        ctts: [observation object from EDGE] Contains data for the CTTS 
        wtts: [observation object from EDGE] Contains data for the WTTS
        F: [list of strings] Energy flux. Should be an array in the form: ['1E+11','1E+12']
        datatag: [str] Tag associated with the spectrum for the ctts, e.g. 'HSTv1'
    
    OPTIONAL INPUTS:
        f: [list of floats] Filling factor. Should be an array in the form: [0.02,  0.001]. Not necessary for the MCMC job
        maskfile: Location of the mask file
        modelpath: path to models
        part_interp: [boolean] If True, interpolate all the model components onto the grid instead of just the Fall. MUCH faster to do this for many values of f.
                               Must be set to true for the MCMC code at the moment.
        
    MCMC INPUTS
        MCMC: Flag to turn on MCMC section of code
        Nruns: Number of MCMC runs
        
    OUTPUTS:
        Returns either a chi2 value or a MCMC walkers, depending on if the MCMC package was run.
        
        
    AUTHOR:
        Connor Robinson, Jul 27, 2016
    
    '''
    
    targ = ctts.name
    
    if f == None:
        f = np.zeros(len(F))
    
    if part_interp == False:
        wlmodel, Fall_model, Fhp_model, Fpre_model, Fphot_model = modelsum(targ, f, F, jobnums, nzeros = nzeros, modelpath = modelpath)
    else:
        wlmodel, Fhp_model, Fpre_model, Fphot_model = modelsum(targ, f, F, jobnums, calc = False, nzeros = nzeros, modelpath = modelpath)
    
    
    #Load in the mask
    maskraw = np.genfromtxt(maskfile, skip_header= 1)
    
    #Sort the data by wavelength + remove nans
    wlsort = np.argsort(ctts.spectra[datatag]['wl'][np.where(ctts.spectra[datatag]['err'] != np.NaN)])
    
    #Now begin to apply the mask to the spectra
    inds = np.array([], dtype = int)
    
    #First mask the areas defined in the mask file
    for region in maskraw:
        inds = np.hstack([inds, np.intersect1d(np.where(ctts.spectra[datatag]['wl'][wlsort]*1e4 > region[0]), np.where(ctts.spectra[datatag]['wl'][wlsort]*1e4 < region[1]))])
    
    #Next mask any data with wavelengths longer or shorter than the wtts spectra
    inds = np.hstack([inds, np.where(ctts.spectra[datatag]['wl'][wlsort]*1e4 < np.min(wlmodel[0]))[0]])
    inds = np.hstack([inds, np.where(ctts.spectra[datatag]['wl'][wlsort]*1e4 > np.max(wlmodel[0]))[0]])
    
    #Also mask where the data is less than zero and smaller than a very small number 
    minflux = 1e-20
    inds = np.hstack([inds, np.where(ctts.spectra[datatag]['lFl'][wlsort] < 0)[0]])
    inds = np.hstack([inds, np.where(ctts.spectra[datatag]['lFl'][wlsort] < minflux)[0]])
    
    #Now apply the mask
    mask = np.ones(len(ctts.spectra[datatag]['lFl']), dtype=bool)
    mask[inds] = False
    
    #Mask data outside the range of the model
    flux_masked = ctts.spectra[datatag]['lFl'][wlsort][mask]
    wl_int = ctts.spectra[datatag]['wl'][wlsort][mask]
    err_int = ctts.spectra[datatag]['err'][wlsort][mask]
    
    #Interpolate model onto data grid
    
    #If the part_interp flag is turned on, interpolate all the model components onto the grid instead
    #of just the Fall. MUCH faster to do this for many values of f.
    
    if part_interp != 0:
        
        Fhp = []
        Fpre = []
        Fphot = []
        
        Fhp_fail = []
        Fpre_fail = []
        Fphot_fail = []
        
        fail = []
        
        for i, F_val in enumerate(F):
            
            funcFhp = interpolate.interp1d(wlmodel[0], Fhp_model[i], kind='linear', bounds_error = False)
            funcFpre = interpolate.interp1d(wlmodel[0], Fpre_model[i], kind='linear', bounds_error = False)
            funcFphot = interpolate.interp1d(wlmodel[0], Fphot_model[i], kind='linear', bounds_error = False)
            
            Fhp.append(funcFhp(wl_int*1e4))
            Fpre.append(funcFpre(wl_int*1e4))
            Fphot.append(funcFphot(wl_int*1e4))
            
            #Check if the code failed anywhere
            Fhp_fail.append(np.where((Fhp[i] <= 0) | (Fhp[i] == np.NaN))[0])
            Fpre_fail.append(np.where((Fpre[i] <= 0) | (Fpre[i] == np.NaN))[0])
            Fphot_fail.append(np.where((Fphot[i] <= 0) | (Fphot[i] == np.NaN))[0])
            
            fail.append(list(set(Fhp_fail[i]).intersection(Fpre_fail[i]).intersection(Fphot_fail[i])))
        
        #Combine the list of all the places where interpolation failed
        all_fail = np.unique(fail)
        
        #Convert arrays to numpy arrays
        Fhp = np.array(Fhp)
        Fpre = np.array(Fpre)
        Fphot = np.array(Fphot)
        
        #Remove the failed points
        if len(all_fail) != 0:
            np.delete(wl_int, all_fail)
            np.delete(Fhp[:,all_fail])
            np.delete(Fpre[:,all_fail])
            np.delete(Fphot[:,all_fail])
            np.delete(flux_masked, all_fail)
            np.delete(err_int, all_fail)
            
        flux = flux_masked
        err = err_int
        
        #Calculate chi2 for each set of filling fraction, simple chi2 analysis here (for grid)
        if MCMC == False:
            #Fix the single value of f cases
            if np.ndim(f) == 0 or np.ndim(f) == 1:
                f = np.array([f])
                
            r_chi2 = []
            for f_val in f:
                Fall = (1-np.sum(f_val)) * Fphot[0] + np.dot(f_val,(Fhp+Fpre))
                
                chi2 = np.sum(((flux - Fall)/err) **2)
                
                #Assume number of points >>> degrees of freedom.
                r_chi2.append(1/len(Fall) * chi2)
                
            if len(r_chi2) == 1:
                r_chi2 = r_chi2[0]
                
            return r_chi2
            
        #using emcee, do a bayesian fit
        if MCMC == True:
            
            ndim, nwalkers = len(F), 100
            pos = [.5 + 1e-1*np.random.randn(ndim) for i in range(nwalkers)]
            
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Fphot, Fhp, Fpre, flux, err), threads = Nthreads)
            sampler.run_mcmc(pos, Nruns)
            
            return sampler
            
        
    if part_interp == 0:
        func = interpolate.interp1d(wlmodel[0], Fall_model, kind='linear', bounds_error = False)
        Fall_int = func(wl_int*1e4)
        
        #Mask where interpolation failed
        nofail = np.where((Fall_int > 0) & (Fall_int != np.NaN))[0]
        
        Fall = Fall_int[nofail]
        wl = wl_int[nofail]
        flux = flux_masked[nofail]
        err = err_int[nofail]
        
        #Calculate chi2
        chi2 = np.sum(((flux - Fall)/err) **2)
        
        #For now just assuming N >> the number of fitted parameters
        r_chi2 = np.array(1/len(Fall) * chi2)
    
        return r_chi2
    
def modelsum(targ, f, F,jobnums,\
    modelpath = '/Users/Connor/Desktop/Research/shock/models/',\
    calc = True, nzeros = 3):
    '''
    shock.modelsum()
    
    PURPOSE:
        Calculates the total shock model
    
    INPUTS:
        targ: [string] Name of the target
        f: Filling factor
        F: Accretion energy flux in ergs/s/cm^2
        
    OPTIONAL INPUTS:
        modelpath: [string] Path to the models. If the default is used, add targ onto the end of the path.
        
    OUTPUTS:
        wl: Wavelenghh
        Fall: The total flux from all the models
        Fhp: Heated photosphere components
        Fpre: Preshock region components
        Fphot: Photosphere components
        
    AUTHOR:
        Connor Robinson Aug 2, 2016
        
    '''
    
    #Fix the modelpath to include the target name if the default path is used
    if modelpath == '/Users/Connor/Desktop/Research/shock/models/':
        modelpath = modelpath+targ+'/'
    
    if len(F) == 1:
        modelname = [targ+'_'+str(jobnums).zfill(nzeros)+'.fits']
        
    else:
        modelname = [targ+'_'+str(job).zfill(nzeros)+'.fits' for job in jobnums]
    
    wl    = []
    Fhp   = []
    Fpre  = []
    Fphot = []
    Ftot  = []
    F_nophot =[]
    
    for i, model in enumerate(modelname):
        data = fits.open(modelpath+model)
        if len(modelname) == 1:
            
            wl = data[0].data[data[0].header['WLAXIS']]
            Fhp = data[0].data[data[0].header['HEATAXIS']]
            Fpre = data[0].data[data[0].header['PREAXIS']]
            Fphot = data[0].data[data[0].header['PHOTAXIS']]
            
            if calc == True:
                Ftot = (1-f)*Fphot+f*(Fhp+Fpre)
                Fall = f*(Fhp+Fpre) + (1-f)*Fphot 
        else:
            
            wl.append(data[0].data[data[0].header['WLAXIS']])
            Fhp.append(data[0].data[data[0].header['HEATAXIS']])
            Fpre.append(data[0].data[data[0].header['PREAXIS']])
            Fphot.append(data[0].data[data[0].header['PHOTAXIS']])
            
            if calc == True:
                Ftot.append((1-f[i])*Fphot[i]+f[i]*(Fhp[i]+Fpre[i]))
                F_nophot.append(f[i]*(Fhp[i]+Fpre[i]))
    
    
    wl    = np.array(wl)
    Fhp   = np.array(Fhp)
    Fpre  = np.array(Fpre)
    Fphot = np.array(Fphot)
    
    if calc == True:
        F_nophot = np.array(F_nophot)
        if len(modelname) != 1:
            f_tot = sum(f)
            Fall = (1-f_tot)*Fphot[0] + np.sum(F_nophot, axis = 0)
            
        return wl, Fall, Fhp, Fpre, Fphot
    else:
        
        return wl, Fhp, Fpre, Fphot

def MCMCsolve(table, ctts, jobs, burnin = 1000, Nruns = 5000, modelpath = '', nzeros = 3, Nthreads = 1):
    '''
    shock.MCMCsolve()
    
    PURPOSE:
        Short function that actually goes about running an MCMC for a set of jobs
    
    INPUTS:
        table: [astropy.ascii table] Table containing all the model information produced by shock_create
        ctts: [EDGE obs file] Classical T Tauri star
        jobs: [int arr/list] Array of job numbers for the models that will be used for fitting
    
    OPTIONAL INPUTS:
        burnin: [int] Number of MCMC steps used for the burn in
        Nruns: [int] Number of MCMC steps
        modelpath: [str] Location of all the models + param files
        nzeros: [int] Zero padding for the jobs
    
    NOTE: All the models MUST have the same veiling + generated from the same dataset for any of this to make sense
    
    AUTHOR:
        Connor Robinson, May 19th, 2017
    
    '''
    targ = ctts.name
    datatag = np.array(table['datatag'][table['jobnum'] == jobs[0]])[0][1:-1]
    
    #Get all the values for F for the set of jobs given
    F = [np.array(table['BIGF'][table['jobnum'] == x])[0][1:-1] for x in jobs]
    
    #Set up the MCMC chain
    sampler = chisqr(ctts, F, jobs, datatag, MCMC = True, Nruns = Nruns, modelpath = modelpath, nzeros = nzeros, Nthreads = Nthreads)
    return sampler.chain[:, burnin:, :].reshape((-1, len(F)))

def mdot(Mctts, Rctts, F, f, Ri = 5):
    '''
    shock.mdot()
    
    PURPOSE:
        Calculates mdot in units of solar masses per year
    
    INPUTS:
        Mctts: [float] Mass of the CTTS in solar masses
        Rctts: [float] Radius of the CTTS in solar radii
        F: [Str Array] Array of flux values
        f: [float array] Array of filling fractions
    
    OPTIONAL INPUTS:
        Ri: [float] Assumed truncation radius in stellar radii (default is 5)
    
    '''
    G = 6.67e-8 #cm^3 g^-1 s^-2
    Msun = 2e33 #g
    Rsun = 6.957e10 #cm
    
    vs = np.sqrt(2*G*(Mctts*Msun)/(Rctts*Rsun)) * np.sqrt(1-1/Ri)
    return 8*np.pi*(Rctts*Rsun)**2/vs**2 * np.sum(np.array([float(x) for x in F]) * f)*(365*24*60*60/Msun)


def cumulativeProb(chi2, veils, xlim = [], save = False, name = None):
    '''
    
    cumulativeProb
    
    PURPOSE:
        Given a set of chi2 values + veilings, produces a cumulative probability plot.
        This plot highlights the most probable value, along with producing percentiles.
    
    INPUTS:
        chi2: [numpy array] chi**2 values for a set of models with EQUALLY spaced values of veiling
        veils: [numpy array] Associated veiling values for the list of chi**2.
    
    OUTPUTS:
        Array containing statistical information structured as follows:
        [mp, sigp1, sigm1]
        
        mp: index of the most probable value
        
        sigp1: index of the lower 16th nd percentil
    Most probable veiling value
    
    
    '''    
    
    if xlim == []:
        xlim = [0,veils[-1]]
    
    prob = np.exp(-np.array(chi2)/2)/np.sum(np.exp(-np.array(chi2)/2))
    
    cProb = np.cumsum(prob)
    
    sigm1 = np.argmin(np.abs(cProb - (0.16)))
    sigm2 = np.argmin(np.abs(cProb - (0.025)))
    
    midpoint = np.argmin(np.abs(cProb - 0.5))
    
    sigp1 = np.argmin(np.abs(cProb - (0.84)))
    sigp2 = np.argmin(np.abs(cProb - (0.975)))
    
    #Set upc colors
    #High light color (purple)
    c1 = '#94DB79' 
    
    #Greens
    c2 = '#27577C' 
    c3 = '#6188A6'
    c4 = '#96B3CC'
    
    #Outline color
    c5 = '#134461'
    
    plt.figure(figsize = [13,7])
    
    #Plot the location of the maximum chi2 value
    probMax = np.argmax(prob)
    plt.bar(veils[probMax]-(veils[1]-veils[0])/2, cProb[probMax], width = veils[1]-veils[0], facecolor = c1, zorder = 2, edgecolor = c3)
    
    #Outline the 50th percentile in white
    plt.bar(veils[midpoint] - (veils[1]-veils[0])/2, cProb[midpoint], width = veils[1]-veils[0], edgecolor = c5, zorder = 3, facecolor = 'none', lw = 2.0)
    
    #Plot the cumulative probability as colored bars
    plt.bar(veils[0:sigm2]-(veils[1]-veils[0])/2, cProb[0:sigm2], facecolor = c2, alpha = .9, width = veils[1]-veils[0], edgecolor = c2,  zorder = 1)
    plt.bar(veils[sigm2:sigm1]-(veils[1]-veils[0])/2, cProb[sigm2:sigm1], facecolor = c3, alpha = .9, width = veils[1]-veils[0], edgecolor = c3,  zorder = 1)
    plt.bar(veils[sigm1:sigp1]-(veils[1]-veils[0])/2, cProb[sigm1:sigp1], facecolor = c4, alpha = .9, width = veils[1]-veils[0], edgecolor = c4,  zorder = 1)
    plt.bar(veils[sigp1:sigp2]-(veils[1]-veils[0])/2, cProb[sigp1:sigp2], facecolor = c3, alpha = .9, width = veils[1]-veils[0], edgecolor = c3, zorder = 1)
    plt.bar(veils[sigp2:]-(veils[1]-veils[0])/2, cProb[sigp2:], facecolor = c2, alpha = .9 , width = veils[1]-veils[0], edgecolor = c2, zorder = 1)
    
    plt.text(veils[sigm1] + xlim[1]*.05, 0.16+ 0.02, r'$r_v = '+str(veils[sigm1])+r'$', fontsize = 17, color = 'w')
    plt.text(veils[midpoint] +xlim[1]*.05, .5+ 0.027, r'$r_v = '+str(veils[midpoint])+r'^{+'+str(veils[sigp1]-veils[midpoint])+'}_{-'+str(veils[midpoint]-veils[sigm1])+'}$', fontsize = 30, color = 'w')
    plt.text(veils[sigp1] +xlim[1]*.05, .84 - 0.05, r'$r_v = '+str(veils[sigp1])+r'$', fontsize = 17, color = 'w')
    
    
    plt.axhline(0.025, ls = '--', color = 'k', alpha = .5, lw = 1)
    plt.axhline(0.16,  ls = '--', color = 'k', alpha = .5, lw = 2)
    plt.axhline(0.5,   ls = '--', color = 'k', alpha = .5, lw = 3)
    plt.axhline(0.84,  ls = '--', color = 'k', alpha = .5, lw = 2)
    plt.axhline(0.975, ls = '--', color = 'k', alpha = .5, lw = 1)
    
    plt.axhline(1.0, color = 'k', alpha = .2, ls = '--', lw = 1)
    
    plt.text(xlim[1]*0.9, 0.01+0.025,  r'$2.5\%$',  color = 'w', fontsize = 15)
    plt.text(xlim[1]*0.9, 0.015+0.16,   r'$16\%$',   color = 'w', fontsize = 17)
    plt.text(xlim[1]*0.9, 0.015+0.5,    r'$50\%$',   color = 'w', fontsize = 25)
    plt.text(xlim[1]*0.9, -0.05+0.84,   r'$84\%$',   color = 'w', fontsize = 17) 
    plt.text(xlim[1]*0.9, -0.04+0.975,  r'$97.5\%$', color = 'w', fontsize = 15) 
    
    plt.xlim(xlim)
    plt.ylim([0, 1.03])
    
    plt.xlabel(r'$r_v$', fontsize = 18)
    plt.ylabel(r'$\int\;P\,(r_v)\,dr_v$', fontsize = 18)
    
    
    if save == True:
        if name != None:
            plt.savefig(name)
        else:
            print('Warning: Specify a name for saving cumulative probability plot.')
    plt.show()
    
    return [veils[sigm2], veils[sigm1], veils[midpoint], veils[sigp1], veils[sigp2]]

def fit_veil(chi2, veils, ctts, datatag, plotpath = '', normpoints = 10000, normmax = 10):
    '''
    fitVeil
    
    PURPOSE:
        Fits for the best value of the veiling using an asymmetric gaussian
    
    INPUTS:
        chi2: [list/numpy array] Chi^2 values from models with different veilings
        veils: [numpy array] Veilings for the models
        ctts: [edge TTS_Obs object] Observation object for the target
        datatag: Tag associated with the data that are being fit
        
    OPTIONAL INPUTS:
        normpoints: [int] Number of points used in the normalization function. Should be fairly large. Default is 10000
        normmax: [float] Maximum veiling used for normalization integration. Default is 10.
    
    OUTPUTS:
        List containing the asymmetric gaussian fit + the normalization with the following structure:
        [Amplitude (A), mean (mu), sigmaL, sigmaR, normalization]
    
    AUTHOR:
        Connor Robinson, June 23rd, 2017
    
    '''
    
    targ = ctts.name
    
    ## Calculate the best value of the veiling based on a chi2 minimization
    #Calculate the probability for each value of veiling
    prob = np.exp(-np.array(chi2)/2)/np.sum(np.exp(-np.array(chi2)/2))
    
    #Fit an antisymmetric gaussian to the probabilities
    p0 = [0.2, .1, .5, .5]
    coeff, var_matrix = optimize.curve_fit(gauss, veils, prob, p0=p0)
    gveils = np.linspace(0,10,10000)
    fit = gauss(gveils, coeff[0], coeff[1], np.abs(coeff[2]), np.abs(coeff[3]))
    
    #Normalize all the distributions
    normfit = 1/(np.sum(gauss(np.linspace(0,normmax,normpoints), coeff[0], coeff[1], np.abs(coeff[2]), np.abs(coeff[3]))) * normmax/normpoints)
    
    #Plot up the probability plot for the veiling
    plt.scatter(veils, prob * normfit, color = 'k', marker = 'o', s = 40, label = 'Data')
    
    #Add shading to show 1, 2 and 3 stdevs
    plt.plot(gveils, fit * normfit, color = 'k', label = 'Gaussian fit', lw = 2, alpha = .5)
    colors = ['b', 'g', 'r']
    stdevs = [0,1,2]
    
    #Loop over each stdev
    for i, stdev in enumerate(stdevs):
        #Create fill regions for right and left sides of gaussian.
        xR = [np.argmin(np.abs(gveils - (coeff[1] + np.abs(coeff[3])*stdev) )), np.argmin(np.abs(gveils - (coeff[1] + np.abs(coeff[3])*(1+stdev)) ))]
        xL =  [np.argmin(np.abs(gveils - (coeff[1] - np.abs(coeff[2])*stdev) )), np.argmin(np.abs(gveils - (coeff[1] - np.abs(coeff[2])*(1+stdev)) ))]
        
        plt.fill_between(gveils[xR[0]:xR[1]+1], fit[xR[0]:xR[1]+1]*normfit, color = colors[i], alpha = .2)
        plt.fill_between(gveils[xL[1]:xL[0]+1], fit[xL[1]:xL[0]+1]*normfit, color  = colors[i], alpha = .2)
    
    #Add the best fit veiling value to the plot
    plt.text(0.7, 0.8, r'$r_v = ' + str(coeff[1])[0:4] + r'^{ +' + str(np.abs(coeff[3]))[0:4]+r'}_{-'+str(np.abs(coeff[2]))[0:4]+r'}$', fontsize = 20)
    
    #Finish off the plot (e.g. ranges, saving ect.)
    plt.ylabel(r'$P\,[r_v]$', fontsize = 17)
    plt.xlabel(r'$r_v$', fontsize = 17)
    plt.ylim([0, 1.0])
    plt.xlim([0, 1.0])
    plt.savefig(plotpath+targ+'_'+datatag+'_veiling_prob.pdf')
    plt.show()
    
    return np.hstack([coeff,normfit])

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
    A, mu, sigmaL, sigmaR = p
    
    y = np.hstack([A*np.exp(-(x[x<mu]-mu)**2/(2.*sigmaL**2)), A*np.exp(-(x[x>=mu]-mu)**2/(2.*sigmaR**2))])
    
    return y


#Define a bunch of functions for doing an emcee fit
def lnlike(f, Fphot, Fhp, Fpre, flux, err):
    Fall = (1-np.sum(f)) * Fphot[0] + np.dot(f,(Fhp+Fpre))
    return np.sum(-0.5*((flux - Fall)/err)**2 - np.log(np.sqrt(2*np.pi) * err))

def lnprob(f, Fphot, Fhp, Fpre, flux, err):
    lp = lnprior(f)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(f, Fphot, Fhp, Fpre, flux, err)

def lnprior(f):
    #Define a top hat between 0 and 1.
    if len(f[f<0]) != 0 or len(f[f>1]) !=0:
        return -np.inf
    return 0.0

