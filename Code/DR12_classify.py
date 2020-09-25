#-*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:22:47 2020

@author: VanesaLara
"""

import astropy.io.fits as fits
from matplotlib import pyplot as plt
import numpy as np
from astropy.table import Table, Column
import sympy as sp
from scipy.integrate import quad
import scipy.integrate as integrate
import math
from scipy.constants import speed_of_light
from astropy.cosmology import FlatLambdaCDM

import astropy.units as u
#from sklearn.descomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans


#--------------------------- DATA PREPARE ---------------------------------------
#Read FITS file DR12
data = fits.getdata('C:/Users/VanesaLara/Desktop/SDSS/DB/DR12Q.fits', 1)
t = Table(data)

#Filtering of redshift quasars
a_rshift = t['Z_VI']
f_rshift = list(filter(lambda dato: dato >= 1 and dato <= 3, a_rshift))
#print(f_rshift[0:10])
#print(len(f_rshift))
#There are 218261 values from full file that have 297301

#------------------------B(z')-------------------

for z in (f_rshift):
    Bz=math.sqrt(0.6889*1+(1-0.6889)*(1+z)**3)
    #print (Bz)
    
#------------------------dL----------------------
"""
#import scipy.integrate as integrate, quad

for i in (f_rshift):
    equ = i/Bz
    sln = integrate (equ,( 0, i))
    print (sln)
    
#error cause "i" should be bring more than one result 

"""
#"""
H0=67.66
for z in (f_rshift):
    dL = ((speed_of_light*(1 + z) /H0)) #i_sln is result of integration
    #print (dL)

#""" 
#error cause "i" is float and the sequence multiply int type, this calling integration function
#------------------------miu----------------------
#"""
const=1
for z in (f_rshift):
    miu = 5*((math.log10(dL)*z)-const)
    #print (miu)     #Taking 73% of hole file size
#"""

#Normalize info
z_max = np.max(f_rshift)
z_min = np.min(f_rshift)

for a in f_rshift:
    norm_rshift = (a-z_min/z_max-z_min)
    #print(norm_rshift)

#Convert list in array
n = np.array(norm_rshift)
na = np.arange(218261).reshape((1, -1))
#print (n.size)
#"""
#-----------------------------EVALUATE ALGORITHMS-----------------------------
wcss = []
for i in range(1, 11) :
	kmeans = KMeans (n_clusters = i, max_iter = 300)
	kmeans.fit(na)
	wcss.append (kmeans.inertia_)
    
#Plot 'Codo de bambú' to check how many clusters could be
plt.plot(range(1, 11), wcss)
plt.title('Codo de Jambú')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

"""
#Apply method
clustering = KMeans(n_clusters = 3, max_inter = 300)
clustering.fit(w_qso)
#Adding classify from original file
z_data['KMeand_Clusters'] = clustering.labels_hdulist[1].data
#PCA to demonstrate how did build clusters
pca = PCA(n_components =2)
pca_qso = pca.fit_transform()

#PCA to demonstrate how did build clusters
pca = PCA(n_components =2)
pca_qso = pca.fit_transform(w_qso)
pca_qso_df = pd.Dataframe(data = pca_qso, columns = ['Componente_1','Componenete_2', 'Componente_3'])

""" 
