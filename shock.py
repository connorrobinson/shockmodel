import itertools
import EDGE as edge
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
import pdb
import matplotlib.patches as mpatches
import os
from scipy import interpolate

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

def scale(targ, wtarg, datatag, veiling, Rwtts, d_wtts, d_ctts,photometry = 0,\
     wttspath = '/Users/Connor/Desktop/Research/shock/data/wtts/',\
     cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/',\
     plotpath = '/Users/Connor/Desktop/Research/shock/plotting/scaled/',\
     outpath  = '/Users/Connor/Desktop/Research/shock/data/wtts/scaled/'):
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
    scaled_spectra = wtts.spectra['HST']['lFl'] * plotfactor
    scaled_photometry = wtts.photometry['Vizier']['lFl'] * plotfactor
    
    #Plot everything up
    plt.plot(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])],ctts.spectra[datatag]['lFl'][np.argsort(ctts.spectra[datatag]['wl'])], color = 'r', label = datatag)
    plt.plot(ctts.photometry['Vizier']['wl'], ctts.photometry['Vizier']['lFl'], 'ro', markersize = 7)
    plt.plot(wtts.spectra['HST']['wl'][np.argsort(wtts.spectra['HST']['wl'])], scaled_spectra[np.argsort(wtts.spectra['HST']['wl'])], color = 'b', label = 'WTTS')
    plt.plot(wtts.photometry['Vizier']['wl'], scaled_photometry, 'bo', markersize=7)
    
    plt.xlabel(r'Wavelength $\rm [\mu m]$')
    plt.ylabel(r'$\rm \lambda F_{\lambda}\,[erg\cdot s^{-1} \cdot cm^{-2}]$')
    
    plt.xlim([1e-1, 1e1])
    plt.ylim([1e-14, 1e-8])
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xscale('log')
    
    plt.savefig(plotpath+targ+'_scaled.eps')
    plt.show()
    
    #Now scale the spectra to the surface of the star
    
    #The code takes in the flux in units of erg s^(-1) cm^(-2) Ang ^(-1)
    #First scale to V band of the CTTS, placing the wtts at the distance of the ctts
    #Then scale to the surface of the star
    #Finally scale for the veiling
    
    factor = (Vflux/wVflux) * ((d_ctts * pc)/(Rwtts * Rsun))**2 * (1/(1+veiling))
    
    flux = (wtts.spectra['HST']['lFl'] / (wtts.spectra['HST']['wl']*1e4)) *  factor
    photometryflux = (wtts.photometry['Vizier']['lFl'] / (wtts.photometry['Vizier']['wl']*1e4)) * factor
    
    #Write everything to a file
    outfile = open(outpath+targ+'_'+wtarg+'.dat', 'w')
    
    for i, item in enumerate(scaled_spectra):
        outfile.write('       ' + str(1e4*wtts.spectra['HST']['wl'][i]).ljust(9,'0') +      '       '+str(flux[i]).zfill(8)+'\n')
    
    #if the photometry flag is turned on, include the photometry in the outputted file. Should only be used for plotting purposes
    if photometry == True:
        for i, item in enumerate(photometryflux):
            outfile.write('       ' + str(1e4*wtts.photometry['Vizier']['wl'][i]).ljust(9,'0') +'       '+str(photometryflux[i]).zfill(8)+'\n') 
        
    outfile.close()
    
    #Make another pickle with the scaled spectra
    wtts_scaled = edge.TTS_Obs(targ+'_'+wtarg)
    
    wtts_scaled.add_spectra('HST', wtts.spectra['HST']['wl'], flux)
    wtts_scaled.add_photometry('HST', wtts.photometry['Vizier']['wl'], scaled_photometry)
    
    #Clean up old pickles
    os.system('rm '+outpath+targ+'_'+wtarg+'_obs.pkl')
    
    #Write a new pickle file
    wtts_scaled.SPPickle(outpath)
    

def create(path,table,names,NAME, wttsfile, samplepath = '', nzeros=3):
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
        if param == 'Teff' or param == 'jobnum':
            continue
        else:
            start = text.find(param + "='")+len(param+"='")
            end = start + len(text[start:].split("'")[0])
            
            if param == 'BIGF':
                text = text[:start] + table[i][1:-1] + text[end:]
            else:
                text = text[:start] + str(table[i]) + text[end:]
                
    
    #Replace the WTTS file
    start = text.find('$DISTANCE\nfort.40\n')+len('$DISTANCE\nfort.40\n')
    end = start + len(text[start:].split("\nEOD")[0])
    text = text[:start] + wttsfile + text[end:]

    
    #Set the name of the file
    start = text.find("NAME='")+len("NAME='")
    end = start + len(text[start:].split("'")[0])
    text = text[:start] + NAME + text[end:]
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the job file
    newjob = open(path+'job'+table[0].zfill(nzeros), 'w')
    newjob.writelines(outtext)
    newjob.close()
    
    ## Write the fort.70 file
    #-------------------------------
    
    #Set up necessary constants
    Tsun = 5780 #K
    Msun = 2e30 #kg
    G    = 6.67e-11 # N * m^2/kg^2
    Rsun = 6.96e8 #m
    
    #Calculate L/Lsun and M/R (in solar units)
    Lsol =  np.round(float(table['RADIO'])**2 * (float(table['Teff'])/Tsun)**4, decimals = 3)
    
    #Calculate the mass to radius ratio
    MR = np.round(float(table['MASS'])/float(table['RADIO']), decimals =3)
    
    #Calculate log(g) in cgs units
    gravity = np.round(np.log10((G * (float(table['MASS'])*Msun)/(float(table['RADIO'])*Rsun)**2) * 100), decimals = 3)
    
    #Write the new file
    newfort = open(path+'fort.70', 'w')
    newfort.writelines(table['Teff']+'   '+str(Lsol)+'    '+table['RADIO']+'    '+table['MASS']+'     '+str(gravity)+'    '+str(MR)+'    0')
    newfort.close()
    

def modelplot(F, f_in, targ, plottarg, wtarg, datatag, dummy_f,\
    wttspath = '/Users/Connor/Desktop/Research/shock/data/wtts/scaled/',\
    cttspath = '/Users/Connor/Desktop/Research/shock/data/ctts/',\
    modelpath ='/Users/Connor/Desktop/Research/shock/models/',\
    plotpath = '/Users/Connor/Desktop/Research/shock/plotting/scaled/'):
    '''
    shock.modelplot
    
    PURPOSE:
        Makes plots of the models + data
    
        Currently can handle 5 models, but more could be added (simply by adding more plotting colors)
    
    INPUTS:
        F: [list of strings] Energy flux. Should be an array in the form: ['1E+11','1E+12']
        f_in: [list of floats] Filling factor. Should be an array in the form: [0.02,  0.001]
        targ: [str] Name of the ctts (the one used in all the filenames ect)
        plottarg: [str] Name of the ctts that will be used for plotting  (e.g. plottarg = 'GM Auriga' while targ = 'gmaur')
        wtarg: [str] Name of the ctts
        datatag: [str] Tag associated with the spectrum for the ctts, e.g. 'HSTv1'
    
    OPTIONAL INPUTS:
        wttspath = path to wtts data. 
        cttspath = path to ctts data
        modelpath = path to models
        plotpath = path for created plot
        
    OUTPUTS:
    
    
    '''
    #Scale f back from the dummy_f used to run the code
    f = f_in
    
    #Fix the modelpath to include the target name if the default path is used
    if modelpath == '/Users/Connor/Desktop/Research/shock/models/':
        modelpath = modelpath+targ+'/'
    
    modelname = ['30.'+x+'.'+targ+'_initial' for x in F]
    
    #Load in the pickles
    ctts = edge.loadPickle(targ,  picklepath = cttspath)
    wtts = edge.loadPickle(targ+'_'+wtarg, picklepath = wttspath)
    
    #Load in the model
    #NOTE: THIS MAY NEED TO CHANGE, NOT SURE IF THE DATA ALWAYS STARTS HERE
    datastart = 119
    footer = 8
    
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
            Ftot = (1-f)*Fphot+f*(Fhp+Fpre)
            Fall = f*(Fhp+Fpre) + (1-f)*Fphot 

        else:
            wl.append(data[:,0])
            Fhp.append(data[:,1]*data[:,0])
            Fpre.append(data[:,2]*data[:,0])
            Fphot.append(data[:,3]*data[:,0])
            Ftot.append((1-f[i])*Fphot[i]+f[i]*(Fhp[i]+Fpre[i]))
            F_nophot.append(f[i]*(Fhp[i]+Fpre[i]))
        
        
    wl    = np.array(wl)
    Fhp   = np.array(Fhp)
    Fpre  = np.array(Fpre)
    Fphot = np.array(Fphot)
    F_nophot = np.array(F_nophot)
    
    if len(modelname) != 1:
        f_tot = sum(f)
        Fall = (1-f_tot)*Fphot[0] + np.sum(F_nophot, axis = 0)
        
    
    #Plot the data
    plt.plot(ctts.spectra[datatag]['wl'][np.argsort(ctts.spectra[datatag]['wl'])] * 1e4,ctts.spectra[datatag]['lFl'][np.argsort(ctts.spectra[datatag]['wl'])], color = 'g')
    plt.plot(ctts.photometry['Vizier']['wl']*1e4, ctts.photometry['Vizier']['lFl'], 'wo', markersize = 5)
    
    #Begin plotting the model
    #Defile colors to be used for the shock. If you want more than 5 components will need to add more colors
    colors = ['b', 'r', 'g', 'm', 'c']
    
    legendhandles = []
    
    if len(modelname) == 1:
        plt.plot(wl, Fhp * f, color = colors[i], alpha = .5, linestyle = '--')#, label = 'Heated Photosphere')
        plt.plot(wl, Fpre * f, color = colors[i], alpha = .5, linestyle = ':')#, label = 'Preshock Region')
        legendhandles.append(mpatches.Patch(color=colors[0], label=r'$F = $'+F[0]+r', $f = $'+str(f)))
    
        plt.plot(wl, Fall, color = 'k', label = 'Total Flux')
        plt.plot(wl, Fphot * (1-f), color = 'k', alpha = .5, label = 'Photosphere')
    
    else:
        for i, model in enumerate(modelname):
            plt.plot(wl[i], Fhp[i] * f[i], color = colors[i], alpha = .5, linestyle = '--')#, label = 'Heated Photosphere')
            plt.plot(wl[i], Fpre[i] * f[i], color = colors[i], alpha = .5, linestyle = ':')#, label = 'Preshock Region')
            legendhandles.append(mpatches.Patch(color=colors[i], label=r'$F = $'+F[i]+r', $f = $'+str(f[i])))
            
        plt.plot(wl[0], Fall, color = 'k', label = 'Total Flux')
        plt.plot(wl[0], Fphot[0] * (1-f[i]), color = 'k', alpha = .5, label = 'Photosphere')
    
    FhpArtist = plt.Line2D((0,1),(0,0), color='k', linestyle='--', label = 'Heated Photosphere')
    FpreArtist = plt.Line2D((0,1),(0,0), color='k', linestyle=':', label = 'Preshock Region')
    FallArtist = plt.Line2D((0,1),(0,0), color='k', label = 'Total Flux')
    PhotArtist = plt.Line2D((0,1),(0,0), color='k', label = 'Photosphere', alpha = .5)
    DataArtist = plt.Line2D((0,1),(0,0), color='g', label = datatag)
    
    legendhandles.append(FhpArtist)
    legendhandles.append(FpreArtist)
    legendhandles.append(FallArtist)
    legendhandles.append(PhotArtist)
    legendhandles.append(DataArtist)
    
    plt.xlabel(r'Wavelength $\rm[\AA]$')
    plt.ylabel(r'$\rm \lambda F_{\lambda}\,[erg \cdot s^{-1} \cdot cm^{-2}]$')  
    
    plt.yscale('log')
    plt.xscale('log')
    
    
    plt.legend(loc = 'best', handles = legendhandles)
    plt.ylim([1e-14, 1e-8])
    plt.xlim([1e3, 2e4])
    plt.title(plottarg)# + r': $f = $'+str(f)+ ', $F =$ ' +modelname[3:3+len(str.split(modelname[3:], '.'+targ)[0])])
    
    plt.savefig(plotpath+targ+'_'+datatag+'.png')
    plt.show()
    
    
    
def chisqr(ctts, wtts, F, f_in, targ, plottarg, datatag, f_dummy,
    maskfile = '/Users/Connor/Desktop/Research/shock/code/mask.dat',\
    modelpath ='/Users/Connor/Desktop/Research/shock/models/',\
    phottag = 'Vizier', wttsphottag = 'Vizier'):
    
    '''
    shock.chisqr
    
    NOTE: CALCULATION OF THE REDUCED CHI^2 MAY BE WRONG.
    
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
        
    OUTPUTS:
        
        
        
    AUTHOR:
        Connor Robinson, Jul 27, 2016
    
    '''
    
    #Scale f back from the dummy_f used to run the code
    f = f_in#/f_dummy
    
    #Fix the modelpath to include the target name if the default path is used
    if modelpath == '/Users/Connor/Desktop/Research/shock/models/':
        modelpath = modelpath+targ+'/'
    
    #Load in the model
    modelname = ['30.'+x+'.'+targ+'_initial' for x in F]
    
    #Get the number of photometry points to trim off the end of the model
    nphot = len(wtts.photometry[wttsphottag]['lFl'])
    
    #NOTE: THIS MAY NEED TO CHANGE, NOT SURE IF THE DATA ALWAYS STARTS HERE
    datastart = 119
    footer = 8 + nphot
    
    wlmodel    = []
    Fhp   = []
    Fpre  = []
    Fphot = []
    Ftot  = []
    F_nophot =[]
    
    for i, model in enumerate(modelname):
        data = np.genfromtxt(modelpath+model, skip_header = datastart, usecols = [1,2,3,4], skip_footer = footer)
        
    #Calculate the total photosphere
        wlmodel.append(data[:,0])
        Fhp.append(data[:,1]*data[:,0])#*1e-8
        Fpre.append(data[:,2]*data[:,0])#*1e-8
        Fphot.append(data[:,3]*data[:,0])#*1e-8
        Ftot.append((1-f[i])*Fphot[i]+f[i]*(Fhp[i]+Fpre[i]))
        F_nophot.append(f[i]*(Fhp[i]+Fpre[i]))
        
    wlmodel    = np.array(wlmodel)
    Fhp   = np.array(Fhp)
    Fpre  = np.array(Fpre)
    Fphot = np.array(Fphot)
    F_nophot = np.array(F_nophot)
    
    f_tot = sum(f)
    Fall_model = (1-f_tot)*Fphot[0] + np.sum(F_nophot, axis = 0)
    
    
    #Load in the mask
    maskraw = np.genfromtxt(maskfile, skip_header= 1)
    
    #Sort the data by wavelength + remove nans
    wlsort = np.argsort(ctts.spectra[datatag]['wl'][np.where(ctts.spectra[datatag]['specErr'] != np.NaN)])

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
    err_int = ctts.spectra[datatag]['specErr'][wlsort][mask]
    
    #Interpolate model onto data grid
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
    r_chi2 = 1/len(Fall) * chi2
    
    
    #plt.loglog(wl, flux, color = 'k')
    #plt.loglog(wl, Fall, color = 'g', alpha = .3)
    
    return r_chi2
    
    
    
    
    
    
