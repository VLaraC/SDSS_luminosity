#-*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:22:47 2020

@author: VanesaLara
"""

import astropy.io.fits as fits
from matplotlib import pyplot as plt
import numpy as np
from astropy.table import Table, Column
from scipy.integrate import quad
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

def Bz (a):
    for z in (a):
        Bz = math.sqrt(0.6889*1+(1-0.6889)*(1+z)**3)
        #print (Bz)
        
        def oper (b):
                iint =lambda x:z/Bz
                sec_eq = (quad(iint, 0, z)[0])
                #print (sec_eq)
                
                def dL (c):
                    H0 = 67.66
                    dL = ((speed_of_light*(1+z)/H0)*sec_eq)
                    #print (dL)
                    
                    def miu (d):
                        miu = 5*((math.log10(dL)*z)-1)
                        print (miu) #Taking 73% of hole file size
                        
                        
                    miu(f_rshift)
                    
                dL(f_rshift)
                
        oper(f_rshift)        
        
Bz(f_rshift)   

"""
#Normalize info
z_max = np.max(miu)
z_min = np.min(miu)

for a in f_rshift:
    norm_rshift = (a-z_min/z_max-z_min)
    print(norm_rshift)
"""    
"""
#Convert list in array
n = np.array(norm_rshift)
na = np.arange(218261).reshape((1, -1))
#print (n.size)
"""
"""
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
"""
#Aplicar método
clustering = KMeans(n_clusters = 3, max_inter = 300)
clustering.fit(w_qso)
#Adding classify from original file
z_data['KMeand_Clusters'] = clustering.labels_hdulist[1].data



#PCA para demostrar como se construyeron los clusters
pca = PCA(n_components =2)
pca_qso = pca.fit_transform(w_qso)
pca_qso_df = pd.Dataframe(data = pca_qso, columns = ['Componente_1','Componenete_2', 'Componente_3'])

""" 
