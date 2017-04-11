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

''''
shock.py

Package that contains most of the code required to work with the Calvet 1998 shock models


FUNCTIONS:
    
    scale: (done)
        Scale WTTS spectra based on veiling + distance + radius of the star
    
    create: (done)
        Creates the shock job file
    
    modelplot: (done)
        Makes plots of the models + data
    
    chisqr:
        Performs chi-squared minimization to get optimum model parameters
    
    
NOTE: Requires EDGE. You can get the latest version at https://github.com/danfeldman90/EDGE
    
'''

def scale(targ, wtarg, datatag, veiling, Rctts, Rwtts, d_wtts, d_ctts,photometry = 0,\
     wttspath = '/Users/Connor/Desktop/Research/shock/data/wtts/',\
     cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/',\
     plotpath = '/Users/Connor/Desktop/Research/shock/plotting/scaled/',\
     outpath  = '/Users/Connor/Desktop/Research/shock/data/wtts/scaled/',\
     wttstag = 'HST', clob = 0):
    '''
     shock.scale
     
     PURPOSE:
        Scale WTTS spectra based on veiling + distance + radius of the star
    
    INPUTS:
         targ: [str] CTTS you are trying to model.
         wtarg: [str] WTTS you are trying to scale.
         datatag: [str] Tag for the CTTS spectra you are comparing against.
         veiling: [float] Veiling at V band. 
         Rwtts: [float] Radius of the wtts in solar units
         d_wtts: [float] Distance to the wtts in pc
         d_ctts: [float Distance to the ctts in pc 
     
     OPTIONAL INPUTS:
         photometry: [boolean] If True, will include photometry in the output file. Only want this for plotting purposes
         wttspath: path to wtts data
         cttspath: path to ctts data
         plotpath: path for plots to end up at
         outpath: path for the scaled spectrum to go
    
    '''
    #Define some constants
    pc = 3.09e16 #m
    Rsun = 6.96e8 #m
    
    #Load in pickles
    wtts = edge.loadPickle(wtarg, picklepath = wttspath)
    ctts = edge.loadPickle(targ,  picklepath = cttspath)
    
    #Find the v band magnitude for the CTTS
    V_wl = 0.545 #V band wavelength in microns
    V_ind =  np.argmin(abs(ctts.photometry['Vizier']['wl']-V_wl))
    Vflux = ctts.photometry['Vizier']['lFl'][V_ind]
    
    wV_ind =  np.argmin(abs(wtts.photometry['Vizier']['wl']-V_wl))
    wVflux = wtts.photometry['Vizier']['lFl'][wV_ind]
    
    #Calculate the flux to scale the spectra
    normflux     = (Vflux/wVflux) / (1 + veiling)
    plotfactor = normflux
    
    #Scale the spectra and the photometry 
    scaled_spectra = wtts.spectra[wttstag]['lFl'] * plotfactor
    scaled_photometry = wtts.photometry['Vizier']['lFl'] * plotfactor
    
    #Plot everything up
    #THIS NEED TO BE WRITTEN IN A BETTER WAY!!!!!
    if datatag != 0:
        plt.plot(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])],ctts.spectra[datatag]['lFl'][np.argsort(ctts.spectra[datatag]['wl'])], color = 'r', label = datatag)
    plt.plot(ctts.photometry['Vizier']['wl'], ctts.photometry['Vizier']['lFl'], 'ro', markersize = 7)
    
    plt.plot(wtts.spectra[wttstag]['wl'][np.argsort(wtts.spectra[wttstag]['wl'])], scaled_spectra[np.argsort(wtts.spectra[wttstag]['wl'])], color = 'b', label = 'WTTS')
    plt.plot(wtts.photometry['Vizier']['wl'], scaled_photometry, 'bo', markersize=7)
    
    plt.xlabel(r'Wavelength $\rm [\mu m]$')
    plt.ylabel(r'$\rm \lambda F_{\lambda}\,[erg\cdot s^{-1} \cdot cm^{-2}]$')
    
    plt.xlim([1e-1, 1e1])
    plt.ylim([1e-14, 1e-8])
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xscale('log')
    
    plt.savefig(plotpath+targ+'_scaled.pdf')
    plt.show()
    
    #Now scale the spectra to the surface of the star
    
    #The code takes in the flux in units of erg s^(-1) cm^(-2) Ang ^(-1) (at the star)
    #Need to scale the observed flux of the WTTS to the surface of the CTTS
    
    factor = (Vflux/wVflux) * ((d_ctts * pc)/(Rctts * Rsun))**2 * (1/(1+veiling))
    
    flux = (wtts.spectra[wttstag]['lFl'] / (wtts.spectra[wttstag]['wl']*1e4)) *  factor
    photometryflux = (wtts.photometry['Vizier']['lFl'] / (wtts.photometry['Vizier']['wl']*1e4)) * factor
    
    #Write everything to a file
    outfile = open(outpath+targ+'_'+wtarg+'veil'+str(veiling)+'.dat', 'w')
    
    for i, item in enumerate(scaled_spectra):
        outfile.write('       ' + str(1e4*wtts.spectra[wttstag]['wl'][i]).ljust(9,'0') +      '       '+str(flux[i]).zfill(8)+'\n')
    
    #if the photometry flag is turned on, include the photometry in the outputted file. Should only be used for plotting purposes
    if photometry == True:
        for i, item in enumerate(photometryflux):
            outfile.write('       ' + str(1e4*wtts.photometry['Vizier']['wl'][i]).ljust(9,'0') +'       '+str(photometryflux[i]).zfill(8)+'\n') 
        
    outfile.close()
    
    #Make another pickle with the scaled spectra
    wtts_scaled = edge.TTS_Obs(targ+'_'+wtarg+'veil'+str(veiling))
    
    wtts_scaled.add_spectra(wttstag, wtts.spectra[wttstag]['wl'], flux)
    wtts_scaled.add_photometry('Vizier', wtts.photometry['Vizier']['wl'], scaled_photometry)
    
    #Write a new pickle file
    wtts_scaled.SPPickle(outpath, clob = clob)
    
    #Plotting for debugging
    # cttsfactor  = ((d_ctts * pc)/(Rctts * Rsun))**2
    # #
    # plt.plot(wtts_scaled.spectra[wttstag]['wl'], wtts_scaled.spectra[wttstag]['lFl'], color = 'b')
    # plt.scatter(wtts_scaled.photometry['Vizier']['wl'], wtts_scaled.photometry['Vizier']['lFl'], color = 'b')
    # #
    # # plt.plot(ctts.spectra[datatag]['wl'], ctts.spectra[datatag]['lFl']/(ctts.spectra[datatag]['wl']*1e4) * cttsfactor, color = 'r')
    # plt.scatter(ctts.photometry['Vizier']['wl'], ctts.photometry['Vizier']['lFl']/(ctts.photometry['Vizier']['wl']*1e4) * cttsfactor, color = 'r')
    # #
    # plt.xscale('log')
    # plt.yscale('log')
    # #
    # plt.show()
    #
    # pdb.set_trace()


def create(path,table,names,NAME, wttsfile, samplepath = '', nzeros=3, \
DIRPROG = '/project/bu-disks/shared/shockmodels/PROGRAMS',\
DIRDAT = '/project/bu-disks/shared/shockmodels/DATAFILES'):

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
        samplepath: Path to the location of the sample. Default is in this directory.
        nzeros: Zero padding in the job number, default is 3
    
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
    #start = text.find('$DISTANCE\nfort.40\n')+len('$DISTANCE\nfort.40\n')
    #end = start + len(text[start:].split("\nEOD")[0])
    
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
    newjob = open(path+'job'+str(table[0]).zfill(nzeros), 'w')
    newjob.writelines(outtext)
    newjob.close()
    
    ## Write the fort.70 file
    #-------------------------------
    
    #Set up necessary constants
    #Tsun = 5780 #K
    #Msun = 2e30 #kg
    #G    = 6.67e-11 # N * m^2/kg^2
    #Rsun = 6.96e8 #m
    
    #Calculate L/Lsun and M/R (in solar units)
    #Lsol =  np.round(float(table['RADIO'])**2 * (float(table['Teff'])/Tsun)**4, decimals = 3)
    
    #Calculate the mass to radius ratio
    #MR = np.round(float(table['MASS'])/float(table['RADIO']), decimals =3)
    
    #Calculate log(g) in cgs units
    #gravity = np.round(np.log10((G * (float(table['MASS'])*Msun)/(float(table['RADIO'])*Rsun)**2) * 100), decimals = 3)
    
    #Write the new file
    #newfort = open(path+'fort.70', 'w')
    #newfort.writelines(table['Teff']+'   '+str(Lsol)+'    '+table['RADIO']+'    '+table['MASS']+'     '+str(gravity)+'    '+str(MR)+'    0')
    #newfort.close()
    

def modelplot(F,jobnums, f_in, targ, plottarg, wtarg, datatag, dummy_f, modeltag,\
    wttspath = '/Users/Connor/Desktop/Research/shock/data/wtts/scaled/',\
    cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/',\
    modelpath ='/Users/Connor/Desktop/Research/shock/models/',\
    plotpath = '/Users/Connor/Desktop/Research/shock/plotting/modelplots/',\
    mask = False, maskfile = '/Users/Connor/Desktop/Research/shock/code/mask.dat',\
    chi2 = -1, nzeros = 3, mdot = -1, xlim = [1e3, 2e4], ylim = [1e-14, 1e-8],\
    photometry = True, spectrim = 0, smooth = 1, plottag = '', loc = 'best'):
    '''
    shock.modelplot
    
    PURPOSE:
        Makes plots of the models + data
    
        Currently can handle 5 models, but more could be added (simply by adding more plotting colors)
    
    INPUTS:
        F: [list of strings] Energy flux. Should be an array in the form: ['1E+11','1E+12']
        jobnum: [list of ints] Job numbers associated with each F (in order)
        f_in: [list of floats] Filling factor. Should be an array in the form: [0.02,  0.001]
        targ: [str] Name of the ctts (the one used in all the filenames ect)
        plottarg: [str] Name of the ctts that will be used for plotting  (e.g. plottarg = 'GM Auriga' while targ = 'gmaur')
        wtarg: [str] Name of the ctts
        datatag: [str] Tag associated with the spectrum for the ctts, e.g. 'HSTv1'
    
    OPTIONAL INPUTS:
        wttspath: [String] path to wtts data. 
        cttspath: [String]path to ctts data
        modelpath: [String] path to models
        plotpath: [String] path for created plot
        mask: [Boolean] If True, will overplot the mask in grey
        maskfile: [String] Path+name of the mask file, default is '/Users/Connor/Desktop/Research/shock/code/mask.dat'
        chi2: [Float] If set to anything other than -1 will print chi**2 on the plot
        photometry: [Boolean] If True, photometry will be included on plot
        spectrim: [list of ints] Pairs of wavelengths to trim the observed between. E.g., [[2100, 2300], [4500,4600]]
        smooth: [Int] Window for smoothing. Automatically set to 1 (no smoothing)
        plottag: [String] Changes the plotting tag from datatag to something else.
        loc: ['String'] Sets the location of your legend
        fontsize:[int]
        
        
    OUTPUTS:
    
    
    '''
    #Scale f back from the dummy_f used to run the code. NOT SURE THIS IS WORKING CORRECTLY!!!!!
    f = f_in
    
    #Fix the modelpath to include the target name if the default path is used
    if modelpath == '/Users/Connor/Desktop/Research/shock/models/':
        modelpath = modelpath+targ+'/'
    
    
    if len(F[0]) == 1:
        modelname = ['fort40.'+targ+str(jobnums).zfill(nzeros)]
    else:
        modelname = ['fort40.'+targ+str(job).zfill(nzeros) for job in jobnums]
        
    #Load in the pickles
    ctts = edge.loadPickle(targ,  picklepath = cttspath)
    wtts = edge.loadPickle(targ+'_'+wtarg, picklepath = wttspath)
    
    #Sum up the components
    wl, Fall, Fhp, Fpre, Fphot = modelsum(targ, modeltag, f, F, jobnums, nzeros = nzeros, modelpath = modelpath)
    
    #Plot the data
    if spectrim == 0:
        plt.plot(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4, pd.rolling_mean(ctts.spectra[datatag]['lFl'][np.argsort(ctts.spectra[datatag]['wl'])],smooth), color = 'g')
    else:
        trimmed = np.array([], dtype = int)
        for region in spectrim:
            trimmed = np.hstack([trimmed, np.intersect1d(np.where(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4 > region[0]), np.where(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4 < region[1]))])
        
        spectrum_trimmed = ctts.spectra[datatag]['lFl'][np.argsort(ctts.spectra[datatag]['wl'])]
#        trimmed_mask[trimmed] = False
        
        #Mask data outside the range of the model
        spectrum_trimmed[trimmed] = np.nan
        
        
        plt.plot(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4, pd.rolling_mean(spectrum_trimmed, smooth), color = 'g')
        
        
    
    
    if photometry == True:
        plt.scatter(ctts.photometry['Vizier']['wl']*1e4, ctts.photometry['Vizier']['lFl'], marker = 'o', s = 30, color = 'g')
    
    #Begin plotting the model
    #Defile colors to be used for the shock. If you want more than 5 components will need to add more colors
    colors = ['b', 'r', 'g', 'm', 'c', 'purple']
    
    legendhandles = []
    
    if len(modelname) == 1:
        plt.plot(wl[np.argsort(wl)], pd.rolling_mean(Fhp[np.argsort(wl)] * f,smooth), color = colors[0], alpha = .5, linestyle = '--', lw = 2)#, label = 'Heated Photosphere')
        plt.plot(wl[np.argsort(wl)], pd.rolling_mean(Fpre[np.argsort(wl)] * f,smooth), color = colors[0], alpha = .5, linestyle = ':', lw = 2)#, label = 'Preshock Region')
        legendhandles.append(mpatches.Patch(color=colors[0], label=r'$F = $'+F+r', $f = $'+str(np.round(f,decimals = 4))))
    
        plt.plot(wl[np.argsort(wl)], pd.rolling_mean(Fall[np.argsort(wl)],smooth), color = 'k', label = 'Total Flux')
        plt.plot(wl[np.argsort(wl)], pd.rolling_mean(Fphot[np.argsort(wl)] * (1-f),smooth), color = 'k', alpha = .5, label = 'Photosphere')
    
    else:
        for i, model in enumerate(modelname):
            plt.plot(wl[i][np.argsort(wl[i])], pd.rolling_mean(Fhp[i][np.argsort(wl[i])] * f[i],smooth), color = colors[i], alpha = .5, linestyle = '--', lw = 2)#, label = 'Heated Photosphere')
            plt.plot(wl[i][np.argsort(wl[i])], pd.rolling_mean(Fpre[i][np.argsort(wl[i])] * f[i],smooth), color = colors[i], alpha = .5, linestyle = ':', lw = 2)#, label = 'Preshock Region')
            legendhandles.append(mpatches.Patch(color=colors[i], label=r'$F = $'+F[i]+r', $f = $'+str(np.round(f[i],decimals = 4))))
            
        plt.plot(wl[0][np.argsort(wl[0])], pd.rolling_mean(Fall[np.argsort(wl[0])],smooth), color = 'k', label = 'Total Flux')
        plt.plot(wl[0][np.argsort(wl[0])], pd.rolling_mean(Fphot[0][np.argsort(wl[0])] * (1-f[i]), smooth), color = 'k', alpha = .5, label = 'Photosphere')
    
    
    if plottag == '':
        plottag = datatag
        
    
    FhpArtist = plt.Line2D((0,1),(0,0), color='k', linestyle='--', label = 'Heated Photosphere', lw = 2)
    FpreArtist = plt.Line2D((0,1),(0,0), color='k', linestyle=':', label = 'Preshock Region', lw = 2)
    FallArtist = plt.Line2D((0,1),(0,0), color='k', label = 'Total Flux')
    PhotArtist = plt.Line2D((0,1),(0,0), color='k', label = 'Photosphere', alpha = .5)
    DataArtist = plt.Line2D((0,1),(0,0), color='g', label = plottag)
    
    
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
    #plt.xscale('log')
    
    
    plt.legend(loc = loc, handles = legendhandles, fontsize = 15)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.title(plottarg)# + r': $f = $'+str(f)+ ', $F =$ ' +modelname[3:3+len(str.split(modelname[3:], '.'+targ)[0])])
    

    plt.savefig(plotpath+targ+'_'+datatag+'.pdf')
    
    plt.show()
    
    
    
def chisqr(ctts, wtts, F, jobnums, targ, plottarg, datatag, f_dummy, modeltag,
    maskfile = '/Users/Connor/Desktop/Research/shock/code/mask.dat',\
    modelpath ='/Users/Connor/Desktop/Research/shock/models/',\
    phottag = 'Vizier', wttsphottag = 'Vizier', part_interp = True, MCMC = False, f_in = 0, Nruns = 2500, nzeros = 3):
    
    '''
    shock.chisqr
    
    PURPOSE:
        Calculates chi-squared for a given model
        
    INPUTS:
        
        ctts: [observation object from EDGE] Contains data for the CTTS 
        wtts: [observation object from EDGE] Contains data for the WTTS
        F: [list of strings] Energy flux. Should be an array in the form: ['1E+11','1E+12']
        f: [list of floats] Filling factor. Should be an array in the form: [0.02,  0.001]
        targ: [str] Name of the ctts (the one used in all the filenames ect)
        plottarg: [str] Name of the ctts that will be used for plotting  (e.g. plottarg = 'GM Auriga' while targ = 'gmaur')
        datatag: [str] Tag associated with the spectrum for the ctts, e.g. 'HSTv1'
        f_dummy: [float] Dummy value of f that was used to run the code so it doesn't crash
    
    OPTIONAL INPUTS:
        maskfile: Location of the mask file
        modelpath: path to models
        phottag: [str] Tag associated with the photometry for the ctts, default is 'Vizier'
        wttsphottag: [str] Tag associated with the photometry for the wtts, default is 'Vizier'
        
        MCMC: Flag to turn on MCMC section of code
        Nruns: Number of MCMC runs
        
    OUTPUTS:
        
        
        
    AUTHOR:
        Connor Robinson, Jul 27, 2016
    
    '''
    
    #Scale f back from the dummy_f used to run the code
    f = f_in#/f_dummy
    
    #Get the number of photometry points to trim off the end of the model
    nphot = len(wtts.photometry[wttsphottag]['lFl'])
    
    if part_interp == False:
        wlmodel, Fall_model, Fhp_model, Fpre_model, Fphot_model = modelsum(targ, modeltag, f, F, jobnums, nphot = nphot, nzeros = nzeros, modelpath = modelpath)
    else:
        wlmodel, Fhp_model, Fpre_model, Fphot_model = modelsum(targ, modeltag, f, F, jobnums, nphot = nphot, calc = False, nzeros = nzeros, modelpath = modelpath)
        
        
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
    
def modelsum(targ, modeltag, f, F,jobnums,\
    nphot = 0, \
    modelpath = '/Users/Connor/Desktop/Research/shock/models/',\
    calc = True, nzeros = 3):
    '''
    shock.modelsum()
    
    PURPOSE:
        Calculates the total shock model
    
    INPUTS:
        targ: [string] Name of the target
        modeltag: [string] tag associated with the models, likely the same as targ
        f: Filling factor
        F: Accretion energy flux in ergs/s/cm^2
        
    OPTIONAL INPUTS:
        nphot: [int] Number of photometry points to trim off the end of the model file
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
        modelname = ['fort30.'+targ+str(jobnums).zfill(nzeros)]
    else:
        modelname = ['fort30.'+targ+str(job).zfill(nzeros) for job in jobnums]
    
#    if len(F[0]) == 1:
#        modelname = ['30.'+F+'.'+modeltag]
#    else:
#        modelname = ['30.'+x+'.'+modeltag for x in F]
    
    #Load in the model
    #NOTE: THIS MAY NEED TO CHANGE, NOT SURE IF THE DATA ALWAYS STARTS HERE
    datastart = 119
    #datastart = 0
    footer = 8 + nphot

    wl    = []
    Fhp   = []
    Fpre  = []
    Fphot = []
    Ftot  = []
    F_nophot =[]
    
    for i, model in enumerate(modelname):
        data = np.genfromtxt(modelpath+model, skip_header = datastart, usecols = [1,2,3,4], skip_footer = footer)
        
        if len(modelname) == 1:
            wl = data[:,0]
            Fhp = data[:,1]*data[:,0]
            Fpre = data[:,2]*data[:,0]
            Fphot = data[:,3]*data[:,0]
            
            if calc == True:
                Ftot = (1-f)*Fphot+f*(Fhp+Fpre)
                Fall = f*(Fhp+Fpre) + (1-f)*Fphot 
        else:
            wl.append(data[:,0])
            Fhp.append(data[:,1]*data[:,0])
            Fpre.append(data[:,2]*data[:,0])
            Fphot.append(data[:,3]*data[:,0])
            
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
    
    
    #if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
    #    return 0.0
    #return -np.inf
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    