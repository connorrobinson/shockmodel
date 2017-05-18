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
    
    
NOTE: Requires EDGE. You can get the latest version at https://github.com/danfeldman90/EDGE
    
'''

def scale(targ, wtarg, datatag, veiling, Rctts, dctts, jobnum, photometry = 0,\
     wttspath = '/Users/Connor/Desktop/Research/shock/data/wtts/',\
     cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/',\
     plotpath = '/Users/Connor/Desktop/Research/shock/plotting/scaled/',\
     outpath  = '/Users/Connor/Desktop/Research/shock/data/wtts/scaled/',\
     wttstag = 'HST', clob = 0, verbose = 1, plot = True, nzeros = 3):
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
         wttspath: [str] path to wtts data
         cttspath: [str] path to ctts data
         plotpath: [str] path for plots to end up at
         outpath: [str] path for the scaled spectrum to go
         clob:[boolean] if true, will overwrite previous file
         verbose: [boolean] if true, code will yell things at you
         plot: [boolean] if true, will make a plot of the scaled photosphere
    
    '''
    #Define some constants
    pc = 3.09e16 #m
    Rsun = 6.96e8 #m
    
    #Fix the job number
    
    jobn = str(jobnum).zfill(nzeros)
    
    #Load in pickles
    wtts = edge.loadPickle(wtarg, picklepath = wttspath)
    ctts = edge.loadPickle(targ,  picklepath = cttspath)
    
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
    
    #Flatten all the photometry into a single array
    # phot = np.array([])
    # photwl = np.array([])
    # for x in ctts.photometry.keys():
    #     phot = np.hstack([phot, ctts.photometry[x]['lFl']])
    #     photwl = np.hstack([photwl, ctts.photometry[x]['wl']])
    #
    # #Get the V band points within maxdiff of the central wl
    # closest = np.abs(photwl - Vwl) < maxdiff
    # Vfluxes = phot[closest]
    # # Now check if there are valid photometry
    # if np.sum(closest) == 0:
    #     raise ValueError('shock.scale ERROR: NO VALID CTTS JOHNSON V BAND PHOTOMETRY. WAVELENGTH MUST BE WITHIN 10nm OF 0.554')
    # #if there are, scale to the mean value of the photometry
    # elif np.sum(closest) == 1:
    #     Vflux = Vfluxes[0]
    # else:
    #     print('MULTIPLE VALUES OF V BAND PHOTOMETRY FOUND FOR CTTS: TAKING MEDIAN VALUE!')
    #     Vflux = np.median(Vfluxes)
    #
    # #Repeat for wtts
    # wphot = np.array([])
    # wphotwl = np.array([])
    # for x in wtts.photometry.keys():
    #     wphot = np.hstack([wphot, wtts.photometry[x]['lFl']])
    #     wphotwl = np.hstack([wphotwl, wtts.photometry[x]['wl']])
    #
    # wclosest = np.abs(wphotwl - Vwl) < maxdiff
    # wVfluxes = wphot[wclosest]
    # # Now check if there  valid photometry
    # if np.sum(closest) == 0:
    #     raise ValueError('shock.scale ERROR: NO VALID WTTS JOHNSON V BAND PHOTOMETRY. WAVELENGTH MUST BE WITHIN 10nm OF 0.554')
    # #if there are, scale to the mean value of the photometry
    # elif  np.sum(closest) == 1:
    #     wVflux = wVfluxes[0]
    # else:
    #     print('MULTIPLE VALUES OF V BAND PHOTOMETRY FOUND FOR WTTS: TAKING MEDIAN VALUE!')
    #     wVflux = np.median(wVfluxes)
    
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
    wtts_scaled.add_photometry('scaled', wphotwl, photflux)
    
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
        
    
def create(path, table, names, NAME, wttsfile, samplepath = '', nzeros=3, \
    DIRPROG = '/project/bu-disks/shared/shockmodels/PROGRAMS',\
    DIRDAT = '/project/bu-disks/shared/shockmodels/DATAFILES',\
    outpath = ''):
    
    '''
    shock.create
    
    PURPOSE:
        Creates the shock job file
    
    INPUTS:
        path: location of the job parameter list
        table: A row from the table containing all of the parameters
        names: The first row in the table containing all of the parameters names.
        NAME: name associated with the model
    
    
    OPTIONAL INPUTS:
        samplepath: [String] Path to the location of the sample. Default is in this directory.
        nzeros: [Int] Zero padding in the job number, default is 3
        DIRPORG/DIRDAT: [String] Paths to where the shockmodels themselves live
        outpath: [String] Path to where the files will be written. Default is the current directory.
        
    
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
        else:
            start = text.find(param + "='")+len(param+"='")
            end = start + len(text[start:].split("'")[0])
            
            if param == 'BIGF':
                text = text[:start] + table[i][1:-1] + text[end:]
            else:
                text = text[:start] + str(table[i]) + text[end:]
                
    
    #Replace the WTTS file
    start = text.find('set filewtts=')+len('set filewtts=')
    end = start +len(text[start:].split('\n')[0])
    text = text[:start] + wttsfile + text[end:]
    
    #Set the name of the file
    start = text.find("NAME='")+len("NAME='")
    end = start + len(text[start:].split("'")[0])
    text = text[:start] + NAME+str(table[0]).zfill(nzeros) + text[end:]
    
    #Set the paths
    start = text.find('DIRPROG=')+len('DIRPROG=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start] + DIRPROG + text[end:]
    
    start = text.find('DIRDAT=')+len('DIRDAT=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start] + DIRDAT + text[end:]
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the job file
    newjob = open(outpath+'job'+str(table[0]).zfill(nzeros), 'w')
    newjob.writelines(outtext)
    newjob.close()
    

def create_runall(jobstart, jobend, clusterpath, outpath = '', samplepath = ''):
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
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the runall file
    newrunall = open(outpath+'runall.csh', 'w')
    newrunall.writelines(outtext)
    newrunall.close()
    

def modelplot(F,jobnums, f, targ, wtarg, datatag,\
    wttspath = '/Users/Connor/Desktop/Research/shock/data/wtts/scaled/',\
    cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/',\
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
        jobnum: [list of ints] Job numbers associated with each F (in order)
        f: [list of floats] Filling factor. Should be an array in the form: [0.02,  0.001]
        targ: [str] Name of the ctts (the one used in all the filenames ect)
        wtarg: [str] Name of the ctts
        datatag: [str] Tag associated with the spectrum for the ctts, e.g. 'HSTv1'
    
    OPTIONAL INPUTS:
        wttspath: [String] path to wtts data. 
        cttspath: [String]path to ctts data
        modelpath: [String] path to models
        plotpath: [String] path for created plot
        mask: [Boolean] If True, will overplot the mask in grey
        maskfile: [String] Path+name of the mask file, default is '/Users/Connor/Desktop/Research/shock/code/mask.dat'
        plottarg: [str] Name of the ctts that will be used for plotting  (e.g. plottarg = 'GM Auriga' while targ = 'gmaur')
        chi2: [Float] If set to anything other than -1 will print chi**2 on the plot
        photometry: [Boolean] If True, photometry will be included on plot
        spectrim: [list of ints] Pairs of wavelengths to trim the observed between. E.g., [[2100, 2300], [4500,4600]]
        smooth: [Int] Window for smoothing. Automatically set to 1 (no smoothing) (Uses the pandas rolling_mean function)
        loc: ['String'] Sets the location of your legend
        fontsize:[int]
        
        
    OUTPUTS:
        Shows + makes a plot of the model
    
    '''
    
    #Fix the modelpath to include the target name if the default path is used
    if modelpath == '/Users/Connor/Desktop/Research/shock/models/':
        modelpath = modelpath+targ+'/'
    
    if len(F[0]) == 1:
        modelname = ['fort40.'+targ+str(jobnums).zfill(nzeros)]
    else:
        modelname = ['fort40.'+targ+str(job).zfill(nzeros) for job in jobnums]
        
    #Load in the pickles
    ctts = edge.loadPickle(targ,  picklepath = cttspath)
    
    if len(np.shape(jobnums)) == 0:
        wtts = edge.loadPickle(targ+'_'+wtarg+'_'+str(jobnums).zfill(nzeros), picklepath = wttspath)
    else:
        wtts = edge.loadPickle(targ+'_'+wtarg+'_'+str(jobnums[0]).zfill(nzeros), picklepath = wttspath)
    
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
    
    
    
def chisqr(ctts, wtts, F, jobnums, targ, datatag, f = None,
    maskfile = '/Users/Connor/Desktop/Research/shock/code/mask.dat',\
    modelpath ='/Users/Connor/Desktop/Research/shock/models/',\
    part_interp = True, MCMC = False, Nruns = 2500, nzeros = 3):
    
    '''
    shock.chisqr
    
    PURPOSE:
        Calculates chi-squared for a given model
        
    INPUTS:
        
        ctts: [observation object from EDGE] Contains data for the CTTS 
        wtts: [observation object from EDGE] Contains data for the WTTS
        F: [list of strings] Energy flux. Should be an array in the form: ['1E+11','1E+12']
        targ: [str] Name of the ctts (the one used in all the filenames ect)
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
    
    if f == None:
        f = np.zeros(len(F))
    
    #Get the number of photometry points to trim off the end of the model
    #nphot = len(wtts.photometry[wttsphottag]['lFl'])
    #wphot = np.array([])
    #wphotwl = np.array([])
    # for x in ctts.photometry.keys():
    #     wphot = np.hstack([wphot, wtts.photometry[x]['lFl']])
    #     wphotwl = np.hstack([wphotwl, wtts.photometry[x]['wl']])
    #
    # nphot = len(wphot)
    
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
            
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wl_int, Fphot, Fhp, Fpre, flux, err))
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
#        modelname = ['fort30.'+targ+str(jobnums).zfill(nzeros)]
        modelname = [targ+'_'+str(jobnums).zfill(nzeros)+'.fits']
        
    else:
        #modelname = ['fort30.'+targ+str(job).zfill(nzeros) for job in jobnums]
        modelname = [targ+'_'+str(job).zfill(nzeros)+'.fits' for job in jobnums]
    
    #Load in the model
    #NOTE: THIS MAY NEED TO CHANGE, NOT SURE IF THE DATA ALWAYS STARTS HERE
#    datastart = 119
    #datastart = 0
#    footer = 8
    
    wl    = []
    Fhp   = []
    Fpre  = []
    Fphot = []
    Ftot  = []
    F_nophot =[]
    
    for i, model in enumerate(modelname):
        #data = np.genfromtxt(modelpath+model, skip_header = datastart, usecols = [1,2,3,4], skip_footer = footer)
        data = fits.open(modelpath+model)
        if len(modelname) == 1:
            # wl = data[:,0]
            # Fhp = data[:,1]*data[:,0]
            # Fpre = data[:,2]*data[:,0]
            # Fphot = data[:,3]*data[:,0]
            
            wl = data[0].data[data[0].header['WLAXIS']]
            Fhp = data[0].data[data[0].header['HEATAXIS']]
            Fpre = data[0].data[data[0].header['PREAXIS']]
            Fphot = data[0].data[data[0].header['PHOTAXIS']]
            
            if calc == True:
                Ftot = (1-f)*Fphot+f*(Fhp+Fpre)
                Fall = f*(Fhp+Fpre) + (1-f)*Fphot 
        else:
            # wl.append(data[:,0])
            # Fhp.append(data[:,1]*data[:,0])
            # Fpre.append(data[:,2]*data[:,0])
            # Fphot.append(data[:,3]*data[:,0])
            
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

#Define a bunch of functions for doing an emcee fit
def lnlike(f, wl, Fphot, Fhp, Fpre, flux, err):
    #m, b, lnf = theta
    #model = m * x + b
    Fall = (1-np.sum(f)) * Fphot[0] + np.dot(f,(Fhp+Fpre))
    #inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return np.sum(-0.5*((flux - Fall)/err)**2 - np.log(np.sqrt(2*np.pi) * err))

def lnprob(f, wl, Fphot, Fhp, Fpre, flux, err):
    lp = lnprior(f)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(f, wl, Fphot, Fhp, Fpre, flux, err)

def lnprior(f):
    #m, b, lnf = theta
    #Define a top hat between 0 and 1.
    if len(f[f<0]) != 0 or len(f[f>1]) !=0:
        return -np.inf
    return 0.0

