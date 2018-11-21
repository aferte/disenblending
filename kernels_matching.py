import numpy as np
import matplotlib.pyplot as plt


'''create p(z)'''
def redshift_distrib(z,z0,alpha,beta): 
    redshift_distrib = z**alpha np.exp(-(z/z0)**beta)
    return redshift_distrib

