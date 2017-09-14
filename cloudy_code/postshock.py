import numpy as np
import matplotlib.pyplot as plt
import pdb
import postshock
import scipy.interpolate as interpolate

'''
postshock.py

PURPOSE:
    Produces emission from the post-shock region for heating the the pre-shock region and the underlying photosphere 
    as part of the Calvet (1998) models. Previously all of this was done by interpolating values from models run for
    a few stars
    
    This code uses the photoionization code Cloudy to calculate volume emissivity + spectra
    
AUTHOR:
    Connor Robinson August 24th, 2017
'''

def dvdz(Lambda, Kappa, g, v, a2):
    '''
    
    postshock.dvdz
    
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
    
    postshock.da2dz
    
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

def structure(Flog, M, R, Ri = 5,\
    ctfile = '/Users/Connor/Desktop/Research/shock/code/cloudy_code/cooling/coolinggrid.txt'):
    
    '''
    
    structure
    
    PURPOSE:
        Integrates downward from the shock to get the post-shock structure
    
     INPUTS:
        Flog:[float] Log10 of the energy flux (F)
        mdot:[float] Mass accretion rate in Msun/yr
        M:[float] Stellar mass in Msun
        R:[float] Stellar radius in Rsun 
    
    OPTIONAL INPUTS:
        Ri:[float] Inner gas disk radius. Default is 5 stellar radii
        ct:[str] Location of the cooling table produce by cloudy
        
    OUTPUTS:
        Array containing z, v, nh, and T.
    
    AUTHOR:
        Connor Robinson, Aug 21st, 2017
    
    '''
    
    #Define constants
    k = 1.380658e-16
    G = 6.67259e-8
    mh = 1.6733e-24
    
    Rsun = 6.96e10
    Msun = 1.99e33
    
    X = 0.70
    Y = 0.28
    Z = 0.02
    
    #Calculate mu. Calvet 1998 uses a pure ionized hydrogen plasma, so using that (for now)
    #Ionized
    mu = (2*X + (3/4)*Y + (1/2)*Z)**(-1)
    #Neutral
    #mu = (X + (1/4)*Y + (1/15.5)*Z)**(-1)
    #Pure ionized hydrogen
    #mu = 0.5
    
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
    
    Kappa = v0*nh0*mu*mh
    
    #Load in the cooling table and interpolate to select Lambda.
    alldata = np.genfromtxt(ctfile, skip_header = 2, dtype = 'float', delimiter = ',')
    LamT = alldata[1:,0]
    Lamnh = alldata[0,1:]
    ct = alldata[1:,1:]
    Lamfunc = interpolate.interp2d(Lamnh, LamT, ct, kind = 'cubic')
    
    Lambda0 = Lamfunc(np.log10(nh0), T0)[0]
    
    #Calculate the velocity derivative at the boundarys
    DVDZ = postshock.dvdz(Lambda0, Kappa, g, v0, a20)
    DA2DZ = postshock.da2dz(Lambda0, Kappa, g, v0, DVDZ)
    
    #Construct grid
    Ncells = 10000
    iterthresh = 1e-4
    
    z = -np.linspace(0,R*Rsun/100000, Ncells)
    dz = z[1]-z[0]
    
    nh = np.zeros(Ncells)
    v = np.zeros(Ncells)
    a2 = np.zeros(Ncells)
    
    nh[0] = nh0
    v[0] = v0
    a2[0] = a20
    
    #Calculate column structure
    for i in np.arange(Ncells-1):
        
        v[i+1] = v[i] + dz * DVDZ
        a2[i+1] = a2[i] + dz * DA2DZ
        nh[i+1] = np.abs(Kappa/(v[i+1]*mu*mh))
        
        #Calculate lambda, left as constant until models finish running
        Lambda = Lamfunc(np.log10(nh[i+1]), (mh*mu)/(k) * a2[i+1])[0]
        
        DVDZ = postshock.dvdz(Lambda, Kappa, g, v[i+1], a2[i+1])
        DA2DZ = postshock.da2dz(Lambda, Kappa, g, v[i+1], DVDZ)
    
    T = mu*mh/k * a2
    
    return np.array([z, v, nh, T])
    

def make_coolingfile(colfiles):
    '''
    
    make_coolingfile
    
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
            data = np.genfromtxt(f, skip_header = 1, usecols=[1, 3])
        except StopIteration:
            print('File empty: ' +f)
            continue
        density = float(f.split('__')[2].split('.col')[0])
        
        outdata.append([data[0], density, data[1]])
    
    outdata = np.array(outdata)
    maxcool = np.max(np.log10(outdata[:,2]))
    mincool = np.min(np.log10(outdata[:,2]))
    
    fig = plt.figure(figsize=(6,6))
    for i, point in enumerate(outdata):
        plt.scatter(np.log10(point[0]), point[1], color = cm.inferno((np.log10(point[2])-mincool)/(maxcool - mincool), 1), s = 50, marker = 's')
    
    sm = plt.cm.ScalarMappable(cmap=cm.inferno, norm=plt.Normalize(vmin=mincool, vmax=maxcool))
    sm.set_array(outdata[:,2])
    plt.colorbar(sm)
    
    plt.show()
    
    #Sort the data into a grid
    unT = np.unique(outdata[:,0])
    unN = np.unique(outdata[:,1])
    
    grid = np.zeros([len(unT), len(unN)])
    for i, x in enumerate(unT):
        for j, y in enumerate(unN):
            ind = (outdata[:,0] == x) * (outdata[:,1] == y)
            if np.sum(ind) == 0:
                grid[i,j] = np.nan
            else:
                grid[i,j] = outdata[ind,2]
    
    x = np.arange(0, grid.shape[1])
    y = np.arange(0, grid.shape[0])
    #mask invalid values
    grid = np.ma.masked_invalid(grid)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~grid.mask]
    y1 = yy[~grid.mask]
    newarr = grid[~grid.mask]

    interp = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                 method='linear')
    
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

    for i, line in enumerate(interp):
        line = str(unT[i]) +', '
        for val in interp[i]:
            line = line +str(val) +', '
        line = line[:-2] + '\n'
        
        gridfile.write(line)
    gridfile.close()
    

def make_coolingmodel(tag,path,T, nh):
    '''
    postshock.make_coolingmodel
    
    PURPOSE:
        Makes cloudy input files for cooling/spectra in the postshock region
    
    INPUTS:
        tag:[string] tag associated with the model. Generally will be 'cooling'
        path:[string] Location to write the cloudy input file
        T:[float] Temperature of the gas !!!!! IN LOG UNITS !!!!!
        nh:[float] Density of the gas !!!!! IN LOG UNITS !!!!!
    
    AUTHOR:
        Connor Robinson Aug 25th, 2017
    '''
    Tname = "{0:.2f}".(T)
    nhname = "{0:.2f}".(nh)
    
    name = tag +'__T'+Tname+ '__n'+ str(nhname)
    newfile = open(path+name+'.in', 'w')
    
    lines = []
    
    lines.append('coronal '+ "{0:.1e}".format(T)+ ' K\n')
    lines.append('hden ' +str(nh)+ '\n')
    lines.append('set dr 0\n')
    lines.append('stop zone 1\n')
    lines.append('cosmic ray background -4\n')
    lines.append('save continuum units microns ".con"\n')
    lines.append('save cooling ".col" no hash last\n')
    
    newfile.writelines(lines)
    newfile.close()
    
    return name

def run_model(name):
    '''
    postshock.run_model
    
    PURPOSE:
        Runs a cloudy model.
    
    INPUTS:
        name:[string] Full name of the model to be run (includes the ".in")
    
    NOTES:
        Must be in the directory with the model
    '''
    
    name_noin = name.split('.in')[0]
    
    infiles_full = glob('*.in')
    outfiles_full = glob('*.out')
    
    print( str(len(outfiles_full))+'/'+str(len(infiles_full)) )
    os.system('cloudy '+name_noin)
    
    
    
    
    
    
    
    
    



