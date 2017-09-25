import numpy as np
import matplotlib.pyplot as plt
import pdb
import prepost
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import os
from glob import glob
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colorbar as cb

'''
prepost.py

PURPOSE:
    Produces emission from the pre + post-shock regions for heating the underlying photosphere 
    as part of the Calvet (1998) models. Previously all of this was done by interpolating values from models run for a few stars
    
    This code uses the photoionization code Cloudy to calculate volume emissivity + spectra
    
AUTHOR:
    Connor Robinson August 24th, 2017
'''

def wrapper(M, R, BIGF, name, ctfile, basepath, cooltag, cloudy, Rstop = 0.1):
    
    '''
    prepost.wrapper
    
    PURPOSE:
        Calls the rest of the code in a single command for use on the cluster.
    
    INPUTS:
        M:[float] Mass of the star in solar units
        R:[float] Radius of the star in solar units
        Flog:[float] Energy flux into the star from accretion (ergs cm^-2 s^1)
        name:[string] Name of the object
        ctfile:[string] Table containing all the cooling coefficients. Has a non-standard structure. 
        basepath:[string] Path to the top level of the directory containing
        cooltag:[string] Tag associated with the cooling models. Default is 'cooling'. Likely do not need to change.
        cloudy:[string] Location of the cloudy executable. 
            
        
    OPTIONAL INPUTS:
        Rstop:[float] Alternative stopping criterion for Cloudy. Either T reaches 4000K or stops at this radius (in stellar units)
    
    OUTPUTS:
        Creates a fort40 file containing the flux and frequency information needed to irradiate the photosphere with the
        following columns: nu, postshock, preshock_in, preshock_out
    
    
    AUTHOR:
        Connor Robinson, September 18th, 2017
    
    '''
    
    #Define constants
    mh = 1.67e-24 #g
    Rsun = 6.96e10 #cm
    c = 3e14 #microns
    
    Ncells = 20000 # Number of cells for postshock structure calculation
    npoints = 100 #Number of locations to produce spectra.
    
    #Convert everything to float
    Flog = np.log10(np.float(BIGF))
    M = np.float(M)
    R = np.float(R)
    
    print('M: '+str(M))
    print('R: '+str(R))
    print('Flog: '+str(Flog))
    print('ctfile: '+ctfile)
    print('basepath: '+basepath)
    print('cooltag: '+cooltag)
    
    #Create the postshock structure
    postshock = prepost.postshock_structure(Flog, M, R, Ncells = Ncells, grid2d = True, logT = True, ctfile = ctfile, dzbase = 1e0)
    
    #Create the postshock spectra
    postshock_spectra = prepost.make_postshockflux(postshock, cooltag, '', name, ctfile = ctfile, basepath = basepath)
    
    #Make the Cloudy model for the preshock
    pre_name = prepost.make_preshockmodel(postshock_spectra, M, R, Flog, name, '', Rstop = 0.1, Ri = 5)
    
    #Call Cloudy
    os.system(cloudy + ' -p '+name)
    
    #Load in the results from Cloudy
    data= np.genfromtxt(name+'.con', usecols = [0, 4, 5], skip_header = 1)
    
    #Convert into the correct units
    nu = c/data[:,0]
    postshock = postshock_spectra[:,1]/nu
    preshock_out = data[:,1]/nu
    preshock_in = data[:,2]/nu
    
    #Write out the fort40 file
    prepost.write_output(nu, postshock, preshock_in, preshock_out, name, fill = 3, outpath = '')
    
    print('Pre-shock and Post-shock regions finished')


def dvdz(Lambda, Kappa, g, v, a2):
    '''
    prepost.dvdz
    
    PURPOSE:
        Calculate dvdz based on the other variables for a single cell.
        v is the velocity and z is the height above the shock.
    
    INPUTS:
        Lambda:[float] Volume emissivity of the plasma
        Kappa:[float] rho * u (Constant, since steady state flow)
        g:[float] Surface gravity. Assumed to be constant since pre-shock region is much smaller than star
        v:[float] Velocity
        a:[float] Sound speed
    
    OUTPUTS:
        dvdz for the cell.
    
    NOTES:
        All units should be in CGS.
        Use the actual mass density, not the number density when calculating Kappa.
    
    AUTHOR:
        Connor Robinson, Aug 22nd, 2017
    
    '''
    dvdz = ((2/5)*Lambda/Kappa - (3/5)*g)/((3/5)*v - a2/v)
    return dvdz

def da2dz(Lambda, Kappa, g, v, dvdz):
    '''
    prepost.da2dz
    
    PURPOSE:
        Calculate da2dz based on other variables for a single cell. Must have a value for da2dz.
        a2 is defined as the isothermal sound speed squared, which can then be used to find the temperature.
        z is the height above the shock.
    
    INPUTS:
        Lambda:[float] Volume emissivity of the plasma
        Kappa:[float] rho * u (Constant, since steady state flow)
        g:[float] Surface gravity. Assumed to be constant since pre-shock region is much smaller than star
        v:[float] Velocity
        dvdz:[float] Derivative of the velocity. Get via postshock.dvdz.
    
    OUTPUTS:
        Derivative of the sound speed squared as a function of height, which can be used to find the temperature
    
    NOTES:
        All units should be in CGS.
        Use the actual mass density, not the number density when calculating Kappa.
    
    AUTHOR:
        Connor Robinson, Aug 22nd, 2017
    '''
    
    da2dz = (-2/5)*v*dvdz-(2/5)*g-(2/5)*Lambda/Kappa
    return da2dz

def postshock_structure(Flog, M, R, Ri = 5, Ncells = 10000, dzbase = 1e3, solver = 'RK',\
    ctfile = 'coolinggrid.txt', logT = True, grid2d = True):
    '''
    prepost.postshock_structure
    
    PURPOSE:
        Integrates downward from the shock to get the post-shock structure
    
     INPUTS:
        Flog:[float] Log10 of the energy flux (F)
        M:[float] Stellar mass in Msun
        R:[float] Stellar radius in Rsun 
    
    OPTIONAL INPUTS:
        Ri:[float] Inner radius of gas disk. Default is 5.
        Ncells:[int] Number of cells in grid to calculate.
        zrange:[float] Region to calculate structure over in stellar radii. Must be large enough to contain entire postshock region.
        solver:[str] Set to either 'euler' or 'RK' to use either a simple 1st order Eulerian scheme or a 4th order Runge-Kutta solver.
        ctfile:[str] Name of the volume emissivity file.
        Tlog:[bool] If true, the code will accept the temperature values for the cooling file in log form. 
        grid2d:[bool] If true, will use the grid with both temperatures and densities. If false, directly accepts .col files from cloudy.
    
    OUTPUTS:
        Array containing z, v, nh, and T.
    
    AUTHOR:
        Connor Robinson, Aug 21st, 2017
    '''
    
    #Define constants (cgs units)
    k = 1.380658e-16
    G = 6.67259e-8
    mh = 1.6733e-24
    
    Rsun = 6.96e10
    Msun = 1.99e33
    
    #Define solar abundances for calculating mean molecular weight (mu)
    X = 0.70
    Y = 0.28
    Z = 0.02
    
    #Calculate/define mu (mean molecular weight). Currently using pure ionized hydrogen flow.
    #Fully Ionized
    #mu = (2*X + (3/4)*Y + (1/2)*Z)**(-1)
    #Neutral
    #mu = (X + (1/4)*Y + (1/15.5)*Z)**(-1)
    #Pure ionized hydrogen
    mu = 0.5
    
    #Calculate relavent quantities
    F = 10**Flog
    vs_preshock = -(2*G*M*Msun/(R*Rsun))**(1/2) * (1-1/Ri)**(1/2)
    g = -G * M*Msun/(R*Rsun)**2
    nh_preshock = np.abs(2*F/(mh*mu*vs_preshock**3))
    
    #Set up boundaries, assuming strong shock
    nh0 = 4*nh_preshock
    v0 = 1/4 * vs_preshock
    T0 = 3/16 * mu*mh/k * vs_preshock**2
    a20 = k*T0/(mu*mh)
    
    #Define Kappa, which will remain constant throughout the column (from continuity equation)
    Kappa = v0*nh0*mu*mh
    
    
    #Load in the cooling table and interpolate to select Lambda.
    if grid2d == True:
        alldata = np.genfromtxt(ctfile, skip_header = 2, dtype = 'float', delimiter = ',')
        LamT = alldata[1:,0]
        Lamnh = alldata[0,1:]
        ct = alldata[1:,1:]
        if logT:
            Lamfunc = interpolate.interp2d(Lamnh, LamT, ct, kind = 'cubic', bounds_error = False)
        else:
            Lamfunc = interpolate.interp2d(Lamnh, np.log10(LamT), ct, kind = 'cubic', bounds_error = False)
    
    if grid2d == False:
        alldata = np.genfromtxt(ctfile, skip_header = 1, usecols = [1,3])
        LamT = alldata[:,0]
        ct = alldata[:,1]
        if logT:
            Lamfunc = interpolate.interp1d(LamT, ct, kind = 'cubic', fill_value = 'extrapolate', bounds_error = False)
        else:
            Lamfunc = interpolate.interp1d(np.log10(LamT), ct, kind = 'cubic', fill_value = 'extrapolate', bounds_error = False)
    
    #Construct grid
    #z = -np.linspace(0,R*Rsun*zrange, Ncells)
    # nh = np.zeros(Ncells)
    # v = np.zeros(Ncells)
    # a2 = np.zeros(Ncells)
    #
    # nh[0] = nh0
    # v[0] = v0
    # a2[0] = a20
    
    nh = [nh0]
    v = [v0]
    a2 = [a20]
    
    z = [0]
    
    #Set the threshold for how much the solution can change between steps before the cell splits.
    thresh = 0.05
    
    #Set the minimum temperature before breaking the loop
    mintemp = 10000
    
    #Initialize some variables for looping
    failcount = 0
    
    dz = -np.abs(dzbase)
    i = 0
    
    #Calculate column structure
    #1st order Eulerian solver CURRENTLY DEFUNCT!!!!
    if solver == 'euler':
        for i in np.arange(Ncells-1):
            
            if grid2d == True:
                Lambda = Lamfunc(np.log10(nh[i]), np.log10((mh*mu)/(k) * a2[i]))[0]
            else:
                Lambda = Lamfunc(np.log10(mh*mu/k) * a2[i]) * nh[i]**2

            DVDZ = prepost.dvdz(Lambda, Kappa, g, v[i], a2[i])
            DA2DZ = prepost.da2dz(Lambda, Kappa, g, v[i], DVDZ)
            
            v[i+1] = v[i] + dz * DVDZ
            a2[i+1] = a2[i] + dz * DA2DZ
            nh[i+1] = np.abs(Kappa/(v[i+1]*mu*mh))
    
    #Dummy value for the sound speed.
    a2out = 1e20
    
    #4th order Runge-Kutta solver
    if solver == 'RK':
        while i < Ncells:
            #Step 1: At t = tn, y = yn
            n1 = nh[i]
            v1 = v[i]
            a2_1 = a2[i]
            
            if grid2d == True:
                L1 = Lamfunc(np.log10(nh[i]), np.log10(mh*mu/k * a2_1))[0]
            else:
                L1 = Lamfunc(np.log10((mh*mu)/(k) * a2_1)) * nh[i]**2
            
            k1 = prepost.dvdz(L1, Kappa, g, v1, a2_1)
            j1 = prepost.da2dz(L1, Kappa, g, v1, k1)
            v2 = v1 + (k1*dz/2) 
            a2_2 = a2_1+ (j1*dz/2)
            
            #Step 2 
            nh2 = np.abs(Kappa/(v2*mu*mh))
            if grid2d == True:
                L2 = Lamfunc(np.log10(nh2), np.log10(mh*mu/k*a2_2))[0]
            else:
                L2 = Lamfunc(np.log10((mh*mu/k)*a2_2)) * nh2**2
            
            k2 = prepost.dvdz(L2, Kappa, g, v2, a2_2)
            j2 = prepost.da2dz(L2, Kappa, g, v2, k2)
            v3 = v1+(k2*dz/2)
            a2_3 = a2_1 + j2*dz/2
            
            #Step 3
            nh3 = np.abs(Kappa/(v2*mu*mh))
            if grid2d == True:
                L3 = Lamfunc(np.log10(nh3), np.log10(mh*mu/k*a2_3))[0]
            else:
                L3 = Lamfunc(np.log10((mh*mu/k)*a2_3)) * nh3**2
            
            k3 = prepost.dvdz(L3, Kappa, g, v3, a2_3) 
            j3 = prepost.da2dz(L3, Kappa, g, v3, k3)
            v4 = v1+(k3*dz)
            a2_4 = a2_1 + j3*dz
            
            #Step 4
            nh4 = np.abs(Kappa/(v4*mu*mh))
            if grid2d == True:
                L4 = Lamfunc(np.log10(nh4), np.log10(mh*mu/k*a2_4))[0]
            else:
                L4 = Lamfunc(np.log10((mh*mu/k) *a2_4))*nh4**2
            
            k4 = prepost.dvdz(L4, Kappa, g, v4, a2_4)
            j4 = prepost.da2dz(L4, Kappa, g, v4, k4)
            
            #Update velocities
            # v[i+1] = v[i] + dz*(k1 + 2*k2 + 2*k3 + k4)/6
            # a2[i+1] = a2[i] + dz*(j1 + 2*j2 + 2*j3 + j4)/6
            # nh[i+1] = np.abs(Kappa/(v[i+1]*mu*mh))
            
            vout = v[i] + dz*(k1 + 2*k2 + 2*k3 + k4)/6
            a2out = a2[i] + dz*(j1 + 2*j2 + 2*j3 + j4)/6
            #nhout = np.abs(Kappa/(v[i+1]*mu*mh))
            nhout = np.abs(Kappa/(vout*mu*mh))
            
            if i == Ncells-1:
                raise ValueError('Maximum number of cells reached before reaching end of postshock region!')
            
            if ((vout - v[i])/v[i] > thresh) or ((vout - v[i])/v[i] > thresh) or ((nhout - nh[i])/nh[i] > thresh) or (np.sum(np.isfinite([vout, a2out, nhout])) !=3) or a2out < 0:
                dz = dz/2 
#                print(i)
#                print(mu*mh/k * a2[i])
                failcount = failcount + 1
                
                if failcount > 100:
                    print('CODE FAILED, TEMPERATURE VALUES LIKELY OUTSIDE COOLING GRID VALUES! RETURNING...')
                    return
                
            else:
                v.append(vout)
                a2.append(a2out)
                nh.append(nhout)
                z.append(z[i]+dz)
                
                failcount = 0
                i = i+1
                dz = dz*2
                
                
                if mu*mh/k * a2out < mintemp:
                    break
    
    z = np.array(z)
    v = np.array(v)
    a2 = np.array(a2)
    nh = np.array(nh)
    
    T = mu*mh/k * a2
    
    return np.array([z, v, nh, T])
    
    

def make_coolingfile(colfiles):
    '''
    prepost.make_coolingfile
    
    PURPOSE:
        Reads in a bunch of outfiles and writes a single parameter file with nh, T and the volume emissivity
    
    INPUTS:
        colfiles:[list/array] All the .col files for the completed runs
    
    OUTPUTS:
        File with nh, T and volume emissivity, Lambda
    
    NOTES: 
        Likely only works if the grid is a rectangle. May want to change this in the future?
        Also will want to remove interpolation section once grid is complete.
    
    '''
    
    outdata = []
    
    for i, f in enumerate(colfiles):
        try:
            data = np.genfromtxt(f, skip_header = 1, usecols=[1,3])
        except StopIteration:
            print('File empty: ' +f)
            continue
        except ValueError:
            pdb.set_trace()
        
        density = float(f.split('__n')[1].split('.col')[0])
        Temperature = float(f.split('__T')[1].split('__')[0])
        
        outdata.append([data[1], density, Temperature])
    
    outdata = np.array(outdata)
    maxcool = np.max(np.log10(outdata[:,0]))
    mincool = np.min(np.log10(outdata[:,0]))
    
    for i, point in enumerate(outdata):
        plt.scatter(point[2], point[1], color = cm.inferno((np.log10(point[0])-mincool)/(maxcool - mincool), 1), s = 50, marker = 's')
    
    sm = plt.cm.ScalarMappable(cmap=cm.inferno, norm=plt.Normalize(vmin=mincool, vmax=maxcool))
    sm.set_array(outdata[:,2])
    plt.colorbar(sm)
    
    plt.show()
    
    #Sort the data into a grid
    unT = np.unique(outdata[:,2])
    unN = np.unique(outdata[:,1])
    
    grid = np.zeros([len(unT), len(unN)])
    
    for i, x in enumerate(unT):
        for j, y in enumerate(unN):
            ind = (outdata[:,2] == x) * (outdata[:,1] == y)
            if np.sum(ind) == 0:
                grid[i,j] = np.nan
            else:
                grid[i,j] = outdata[np.where(ind)[0][0],0]
                
    
    #Write out the grid + temp/density
    gridfile = open('coolinggrid.txt', 'w')
    gridfile.write('Grid containing volume emissivities from Cloudy coronal models. First row is log10(nh), first column is T\n')
    gridfile.write('#-------------------------------------------------------------------------------------------------------#\n')
    
    gridfile.write('0, ')
    line = ''
    for val in unN:
        line = line + str(val) +', '
    line = line[:-2] +'\n'
    gridfile.write(line)

    for i, line in enumerate(grid):
        line = str(unT[i]) +', '
        for val in grid[i]:
            line = line +str(val) +', '
        line = line[:-2] + '\n'
        
        gridfile.write(line)
    gridfile.close()
    

def make_coolingmodel(tag,path,T, nh):
    '''
    prepost.make_coolingmodel
    
    PURPOSE:
        Makes cloudy input files for cooling/spectra in the postshock region
    
    INPUTS:
        tag:[string] tag associated with the model. Generally will be 'cooling'
        path:[string] Location to write the cloudy input file
        T:[float] Temperature of the gas !!!!! IN LOG10 UNITS !!!!!
        nh:[float] Density of the gas !!!!! IN LOG10 UNITS !!!!!
    
    AUTHOR:
        Connor Robinson Aug 25th, 2017
    '''
    
    Tname = "{0:.2f}".format(T)
    nhname = "{0:.2f}".format(nh)
    
    name = tag +'__T'+Tname+ '__n'+ str(nhname)
    newfile = open(path+name+'.in', 'w')
    
    lines = []
    
    lines.append('coronal ' + "{0:.2e}".format(10**T) + ' K\n')
    lines.append('hden ' +str(nh)+ '\n')
    lines.append('set dr 0\n')
    lines.append('stop zone 1\n')
    lines.append('cosmic ray background -4\n')
    lines.append('save continuum units microns ".con" no hast last\n')
    lines.append('save cooling ".col" no hash last\n')
    lines.append('iterate to convergence')
    
    newfile.writelines(lines)
    newfile.close()
    
    return name

def run_model(name):
    '''
    prepost.run_model
    
    PURPOSE:
        Runs a cloudy model.
    
    INPUTS:
        name:[string] Full name of the model to be run (includes the ".in")
    
    NOTES:
        Must be in the directory with the model
    '''
    
    name_noin = name.split('.in')[0]
    
    #Get the directory name
    directory = ''
    for x in name.split('/')[1:-1]:
        directory = directory + '/' + x
    directory = directory + '/'
    
    #Get the name that cloudy needs to run
    run_name = name.split(directory)[1].split('.in')[0]
    
    print(run_name)
    
    #Run cloudy
    os.system('cloudy '+run_name)
    
def collect_models(basepath = '/Users/Connor/Desktop/Research/shock/code/cloudy_code/models/'):
    '''
    prepost.collect_models
    
    PURPOSE:
        Collect and return all of the cooling models that have finished running
    
    INPUTS:
        None
    
    OPTIONAL INPUTS:
        basepath:[string] Top level of where the model files are kept
    
    OUTPUTS:
        Returns a list of all the models that have been completed
    '''
    
    directories = glob(basepath+'n_*')
    
    col = []
    
    for d in directories:
        files = glob(d+'/'+'*.col')
        for f in files:
            if os.stat(f).st_size == 0:
                print('Empty file: ' + f)
            else:
                col.append(f)
    
    return col
    
    
def make_postshockflux(model, cooltag, specpath, spectag,\
            ctfile = '/Users/Connor/Desktop/Research/shock/code/cloudy_code/coolinggrid.txt',\
            basepath = '/Users/Connor/Desktop/Research/shock/code/cloudy_code/models/'):
    '''
    prepost.make_flux
    
    PURPOSE:
        Calculates and saves the emergent spectrum from the given post-shock structure
    
    INPUTS:
        npoints:[int]  Number of points to interpolate structure over
        model:[array] Structure array from postshock.structure
        cooltag:[string] Tag associated with the cooling file
        specpath:[string] Path where the spectra will be written. Name is generated automatically
        spectag:[string] Tag associated with the object being modeled.
        
    OPTIONAL INPUTS:
        basepath:[string] Top level of where the model files are kept
        ctfile:[string] File containing information about available T and rho pairs. Made by postshock.make_coolingfile
    
    OUTPUTS:
        The total spectrum produced by the post-shock region, and a file containing the spectra + wl
    
    NOTES:
        The spectrum that is written out does not contain bins with zero flux while the returned spectra does.
    
    '''
    
    c = 2.998e8 #m/s
    q = 1.6e-19 #C
    h = 6.63e-34 #J s 
    
    #Load in the cooling table
    alldata = np.genfromtxt(ctfile, skip_header = 2, dtype = 'float', delimiter = ',')
    gridT = alldata[1:,0]
    gridnh = alldata[0,1:]
    
    #If there are nans, look at the structure that is not nan
    if np.sum(~np.isfinite(model[3,:])) != 0:
        cut = np.where(~np.isfinite(model[3,:]))[0][0]-1
        print('Warning in prepost.make_flux: Careful! NaNs detected')
    
    Z = model[0,:]
    NH = model[2,:]
    T = model[3,:]
    
    print('Number of valid cells: '+str(len(Z)))
    
    #For each point in model, find the closest model I have, and make a stack of spectra
    sstack = []
    for i, z in enumerate(Z):
        
        #Useful things for testing.
        #print(str(i+1) + '/'+str(len(Z)))
        #print(str(i)  +': '+ str((10**Tval - T[i])/T[i]))

        if i !=0:
            dz = np.abs(Z[i]-Z[i-1])
        else:
            dz = np.abs(Z[i+1]-Z[i])
        
        
        Tval = gridT[np.argmin(np.abs(np.log10(T[i]) - gridT))]
        nhval = gridnh[np.argmin(np.abs(np.log10(NH[i]) - gridnh))]
        
        Tname = "{0:.2f}".format(Tval)
        nhname = "{0:.2f}".format(nhval)
        
        
        #Load in the spectra, get the wavelength and total emission.
        full = np.genfromtxt(basepath+'n_'+nhname[:-1]+'/'+cooltag+'__T'+Tname+'__n'+nhname+'.con', usecols = [0,6])
        
        wl = full[:,0]
        s = full[:,1]
        
        #Separate out last iteration
        if np.sum(np.diff(wl) > 0) !=0:
            cut = np.where((np.diff(wl) > 0))[0][-1]+1
            s = full[cut:,1]
            wl = full[cut:,0]
            
        sstack.append(s * dz)
        
    #Get the wavelength information from the final column
    sstack = np.array(sstack)
    
    #Sum each block. Divide by 2, since I am taking column 7 which is the total emission emitted by each cell
    #Units from Cloudy are erg cm^-2 s^-1. Since working with 1cm x 1cm column, this is fine.
    spectra_all = np.sum(sstack, axis = 0)/2
    
    #Remove the places where the flux is 0.
    non_zero = spectra_all != 0
    spectra = spectra_all[non_zero]
    
    wlnz = wl[non_zero]
    
    sfile = open(specpath+spectag+'.sed', 'w')
    sfile.write("{0:.6e}".format(wlnz[0]) +'\t'+ "{0:.6e}".format(spectra[0]) + '\t'+'units microns\n')
    
    for i in np.arange(len(spectra[1:]))+1:
        sfile.write("{0:.6e}".format(wlnz[i]) +'\t'+ "{0:.6e}".format(spectra[i])+'\n')
    
    sfile.close()
    
    return np.transpose(np.vstack([wl, spectra_all]))
    

def preshock_structure(Flog, M, R, Ri = 5):
    '''
    prepost.structure
    
    PURPOSE:
        Computes the structure of the pre-shock region. Assumes constant density + gravity (extent is small compared to star)
    
    INPUTS:
        Flog:[float] Log10 of the energy flux (F)
        mdot:[float] Mass accretion rate in Msun/yr
        M:[float] Stellar mass in Msun
        R:[float] Stellar radius in Rsun=
    
    OPTIONAL INPUTS:
        Ri:[float] Inner gas disk radius. Default is 5 stellar radii
    
    OUTPUTS:
        The density of the pre-shock region.
    
    AUTHOR:
        Connor Robinson, August 29th, 2017
    '''
    
    #Define constants
    G = 6.67259e-8
    mh = 1.6733e-24
    
    Rsun = 6.96e10
    Msun = 1.99e33
    
    #Calculate mu. Calvet 1998 uses a pure ionized hydrogen plasma. Using that for now.
    X = 0.70
    Y = 0.28
    Z = 0.02
    #Ionized
    #mu = (2*X + (3/4)*Y + (1/2)*Z)**(-1)
    #Neutral
    #mu = (X + (1/4)*Y + (1/15.5)*Z)**(-1)
    #Pure ionized hydrogen
    mu = 0.5
    
    F = 10**Flog
    vs_preshock = -(2*G*M*Msun/(R*Rsun))**(1/2) * (1-1/Ri)**(1/2)
    nh_preshock = np.abs(2*F/(mh*mu*vs_preshock**3))
    
    return nh_preshock

def make_preshockmodel(postshock_spectra, M, R, Flog, tag, path, Rstop = 0.1, Ri = 5):
    '''
    prepost.make_preshockmodel():

    PURPOSE:
        Makes cloudy input files for the pre-shock region

    INPUTS:
        nh:[float] Number density  of the gas !!!!! IN LOG10 UNITS !!!!!
        Flog:[float] Energy flux into the star per cm^2 (rho * u^3)
        tag:[string] Name associated with the file
        path:[string] Location of spectra file, and where the cloudy input file will be created. This is the same place by necessity.
        R:[float] Stellar radii in solar units
        Lpost:[float] Intensity integrated over frequency. With perfect ODE solutions, this should be F/2
        
    OPTIONAL INPUTS:
        Rstop:[float] Height above shock to stop running code if 4000k is not reached first.
    
    
    AUTHOR:
        Connor Robinson, Aug 29th, 2017
    '''
    
    #Integrate the postshock spectra
    Lpost = np.abs(integrate.trapz(postshock_spectra[:,1]/postshock_spectra[:,0], postshock_spectra[:,0]))
    
    #Calculate the preshock density
    nh = prepost.preshock_structure(Flog, M, R, Ri = Ri)
    
    Rsun = 6.96e10 #cm
    
    Rstop = 0.1
    
    nhname = "{0:.2f}".format(nh)
    Flogname = "{0:.2f}".format(Flog)
    
    #Load in the spectral file and integrate between two wavelengths to get the intensity
    spectra = np.genfromtxt(path+tag+'.sed', usecols = [0,1])
    
    c = 3e14 #microns
    
    nu = c/spectra[:,0]
    Fnu = spectra[:,1]/nu
    Lum = integrate.trapz(Fnu, nu)
    
    
    # scale_wl = 0.001 #microns
    # scale_ind = np.argmin(np.abs(scale_wl-spectra[:,0]))
    # scale_wl_model = spectra[scale_ind,0]
    # scale_flux = spectra[scale_ind,1]
    
    #Scale the incident flux by the emission from the postshock
    scale = np.log10(Lpost)
    
    
    name = tag+'.in'
    newfile = open(path+name, 'w')
    
    lines = []
    
    #Constant density
    lines.append('hden ' +"{0:.5}".format(np.log10(nh))+ '\n')
    #Set distance to be far away from star (approximates plane parallel)
    lines.append('radius 18 \n')
    #Define the shape of the incident radiation
    lines.append('table SED "'+tag+'.sed" \n')
    #Scale the incident radiation
    lines.append('Intensity '+"{0:.5}".format(scale)+'\n')
    #lines.append('nuf(nu) = '+"{0:.4f}".format(np.log10(scale_flux)) + ' at '+"{0:.7f}".format(scale_wl_model) +' microns')
    #Set the stop criterion at 4000K
    lines.append('stop temperature 4e3 K \n')
    #Set an additional stop command. Stop the code at 10% of the stellar radii if it hasn't reached 4000K yet.
    lines.append('stop depth '+str(np.log10(Rsun*R*Rstop)))
    #Cosmic rays. Would expect them to be quite a bit weaker than ISM?
    lines.append('cosmic ray background -4 \n')
    #Save the continuum
    lines.append('save continuum units microns ".con" \n')
    #Save the temperature/hydrogen ionization
    lines.append('save overview ".ovr" no hash \n')
    
    newfile.writelines(lines)
    newfile.close()
    
    return name

def write_output(nu, postshock, preshock_in, preshock_out, name, fill = 3, outpath = ''):
    '''
    prepost.write_output
    
    PURPOSE:
        Write the final output to be used in columna15
    
    INPUTS:
        nu:[array] Frequency values in hz
        postshock:[array] Emission from postshock region. Should be half of the total emission from this region. 
        preshock_in:[array] Inward flux from pre-shock region
        preshock_out:[array] Outward flux from pre-shock region
        name:[string] Name of the object
    
    OPTIONAL INPUTS:
        fill:[int] Amount of zero padding, default is 3
        outpath:[string] Output location for the file
    
    OUTPUTS:
        fort40 file containing the following columns: nu, postshock, preshock_in, preshock_out
    
    NOTES:
        Input Units: ergs cm^-2 s^-1 Hz^-1
        However, output from cloudy: ergs cm^-2 s^-1 (Shouldn't need to multiply by any constants)
    
    AUTHOR:
        Connor Robinson, September 14th, 2017
    '''
    
    
    # Format for the name: fort40.name
    name = 'fort40.'+name
    
    # Format for writing out: nu, postshock, preshock-in, preshock-out, no header
    data = np.transpose(np.vstack([nu, postshock, preshock_in, preshock_out]))
    
    # Only keep nonzero fluxes
    good = (data[:,1] > 0) * (data[:,2] >0) * (data[:,3] > 0)
    
    np.savetxt(outpath+name, data[good], fmt='%.4e')
    
    
    
    
    
    
    
