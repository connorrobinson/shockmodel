import prepost
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

'''
make_grid.py

PURPOSE:
    Script that makes a grid of cloudy models and runs them using the framework of the postshock module

AUTHOR:
    Connor Robinson, August 28th, 2017
'''

#Switch for turning off model creation
make = True

tag = 'cooling'
basepath = '/Users/Connor/Desktop/Research/shock/code/cloudy_code/models/'
T = np.linspace(3.7, 7, 34)
#nh = np.linspace(15.2, 16, 9)
nh = np.linspace(10.0, 17, 71)


models = []

#Make the models
if make == True:
    for i in np.arange(len(T)):
        for j in np.arange(len(nh)):
            
            #Construct a directory if none exists
            directory = 'n_'+str(nh[j])
            if os.path.isdir(basepath+directory) == False:
                os.system('mkdir '+basepath+directory+'/')
            
            name = prepost.make_coolingmodel(tag,basepath+directory+'/',T[i], nh[j])
            models.append(basepath+directory+'/'+name+'.in')


models = np.array(models).astype('str')

#Write out the list of models 
modelfile = open(basepath+'T'+str(T[0])+'-'+str(T[-1])+'__n'+str(nh[0])+'-'+str(nh[-1])+'__inmodels.txt', 'w')
for line in models:
    modelfile.write(line+'\n')
modelfile.close()


todo = []
paths = []
#Find all the models that have not already been run
for m in models:
    if os.path.exists(m.split('.in')[0]+'.out') == False:
        todo.append(m)
        paths.append(m.split(tag+'__')[0])
        

todo = np.array(todo)
paths = np.array(paths)

#Get the different directories for each density
unP = np.unique(paths)

#Run each model in parallel
for i in np.arange(len(unP)):
    os.chdir(unP[i])
    pool = Pool()
    batch = todo[paths == unP[i]]
    pool.map(prepost.run_model,batch)

#Construct the cooling grid
grids = ['T3.7-7.0__n10.0-17.0__inmodels.txt']#, \
#         'T3.5-3.6__n10.0-17.0__inmodel.txt']
#grids = ['T3.5-7.0__n14.0-16.0__inmodels.txt',\
#         'T3.7-7.0__n15.2-16.0__inmodels.txt',\
#         'T3.5-7.0__n10.0-14.0__inmodels.txt',\
#         'T3.7-7.0__n16.1-17.0__inmodels.txt']

colfiles = prepost.collect_models('col', basepath=basepath)
colfiles = np.array(colfiles)
prepost.make_coolingfile(colfiles)

opcfiles = prepost.collect_models('opc', basepath=basepath)
opcfiles = np.array(opcfiles)
prepost.make_opacityfile(opcfiles)



 



