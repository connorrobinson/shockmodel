------------------------------------------------------------
Notes on using shock.py + associated scripts (shock_create.py and shock_analysis.py)

shock.py is a module that contains functions that are useful for the analysis of the Calvet (1998) shock models. 

Connor Robinson
------------------------------------------------------

Intro:
Unlike EDGE, there are no 'utility' functions (e.g. EDGE.convertMag) in shock.py, and will rely on the code already present
in EDGE to handle this. The scripts that are in the Github have very general calls to the functions in shock.py, so the only
parameters that you should need to change on a regular basis are those at the top of the two scripts.

This code requires David Forman-Mackey's 'emcee' and 'corner' modules in order to run. You may have to install them. 


shock_create
------------------------------------------------------
This script is similar to jobmaker.py in that it takes lists of the model parameters and creates the job files.
In addition to this, it also creates the scaled WTTS spectra files that are required for the code to run using 
the shock.scale function.

Parameters that need to be changed:
------------------

path: Path to where the jobfiles should be sent on your local machine
clusterpath: Path on the cluster where you will be running your jobs
samplepath: Path to the location of the sample job file
wttspath: Path to where the WTTS pickles are stored
cttspath: Path to where the CTTS pickles are stored

NAME: Name of the object. 
datatag: EDGE observation tag associated with a given observation. I have been using HST or HSTv1, HSTv2 ect. for mine.
wtarg: Name of the associated WTTS. For the 4 that I have been working with, they are: recx1, hbc427, lkca19 and twa7.

DISTANCE: distance in pc
VEILING: Veiling. I have been doing between 0 and 1 in increments of 0.01 for solving for the veiling
MASS: Mass in solar masses
RADIO: Radius in solar radii
BIGF: Energy flux in erg s^-1 cm^-2. Must be entered in the following format: '1E+10', '3E+11', ect.
TSTAR: Effective temperature of the star in K

Nzeros: Zero padding on job numbers. 4 should be enough for most cases.

You should not need to change anything else in this file, unless you are running the code in a different directory on the cluster than the default
The paths for that are specified by the DIRPROG and DIRDAT variabiles.


Once this code has been run, transfer + the models + the scaled WTTS spectra to the cluster and run them. 
Note that the code automaticallygenerates a runall.csh file in addition to the job files, and a file 
containing information about each of the models in the grid.


collate
------------------------------------------------------
Once the jobs have finished, collate them using the 'shock = 1' flag. collate.masscollate works for the shock models and is generally
the best option for collating large amounts of files.

Once they have been collated, transfer them back to your local machine.

shock_analysis
------------------------------------------------------
This code does all analysis for solving for the best fitting values of the veiling alongside the filling fractions.

The filling fractions for each flux are found using an MCMC simulation, which also produces errors along with a triangle plot. 
The veiling is found by running this MCMC simulation for the filling fractions for each value of the veiling and then finding the
minimum chi^2 in the list of veilings.

This entire process is repeated for each HST visit. 

The script will produce three plots, the veiling probability distribution function, the corner plot. 
The inputs for this are similar to shock_create, but somewhat simplified since the model parameters are grabbed from the parameter file.

This code is set up to run in parallel, since the MCMC simulations are somewhat costly (~45s each on my laptop). Averaging over the
entire list of veiling values, parallization reduces this to ~15s per run on my laptop with 4 cores. On a computer with more cores 
available, this should decrease even further.

Parameters that need to be changed
------------------

targ: Name of the object, (same as NAME parameter from shock_create)
wtarg = Name of the associated WTTS object
tags = EDGE observation tags. Same as datatags from before.
cttspath = Path to the CTTS pickles
modelpath = Path to the .fits files from collating the jobs.

The next parameters you can change, but it is less likely that you will need to change them between different objects.

The defaults are probably ok for mos o
nzeros: Zero padding, 4 is probably file
Nruns: MCMC interations. 5000 seems to work ok. This can be increased but will increase the amount of time the model takes
burnin: MCMC burn in. 1000 seems to work ok, but can be increased at the cost of some iterations.
ncores: My laptop has 4 cores, so I have assigned all 4 cores to work on it (essentially freezing my computer while it runs)
        If your computer has more cores you may want to increase this. 
Ri: Inner edge of the disk




