import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from astropy import cosmology
from astropy import constants as const

'''cosmology'''
H0,Om = 68.3, 0.299
cosmo = cosmology.FlatLambdaCDM(H0,Om) 

def pz(z,z0 = 0.5,alpha= 1., beta = 2.): 
    '''create redshift distribution'''
    redshift_distrib = z**alpha *np.exp(-(z/z0)**beta)
    return redshift_distrib


def lensing_kernel(z1,z2):
    '''compute lensing kernel'''
    factor = const.c**2 / (4*np.pi*const.G)
    sig_crit = factor * cosmo.angular_diameter_distance(z2) / cosmo.angular_diameter_distance(z1) /  cosmo.angular_diameter_distance_z1z2(z1,z2)
    sig_crit_inv = 1./sig_crit.cgs.value
    sig_crit_inv[~np.isfinite(sig_crit_inv)] = 0.
    sig_crit_inv[sig_crit_inv < 0.]= 0.
    return sig_crit_inv


def g_kernel(z,zmax=3,ngrid=1000):
    '''integral of the lensing kernel '''
    
    '''z to ingrate on'''
    zgrid = np.linspace(0,zmax,ngrid)
    pzgrid = pz(zgrid)
    if len(z) > 1:
        result = np.zeros_like(z)

        for i,thisz in enumerate(z):
            '''integral over z greater than z of evaluation'''
            integrand = lensing_kernel(thisz,zgrid[zgrid>thisz]) * pzgrid[zgrid>thisz]
            result[i] = np.trapz(integrand,x=zgrid[zgrid>thisz])
        return result
    else:
        integrand = lensing_kernel(thisz,zgrid)*pzgrid
        return np.trapz(integrand,x=zgrid)

def g_kernel_blend(zl,zmax=3,ngrid=1000,R1 = 0.7,R2 = 0.3):
    '''integral of the lensing kernel for 2 blended galaxies'''
    
    '''z to ingrate on'''
    zgrid = np.linspace(0,zmax,ngrid)    
    if len(zl)>1:
        result = np.zeros_like(zl)
        '''redshift distribution for each galaxy'''
        pz1_integrand = pz(zgrid) 
        pz2_integrand = pz(zgrid)
        for i,zli in enumerate(zl):
            zbgrid1,zbgrid2 = np.meshgrid(zgrid,zgrid)
            '''lensing kernel for each galaxy for redshift of kernel evaluation. Array of size the integrand'''
            sigcrit_inv_1 = lensing_kernel(zli,zbgrid1[0,zbgrid1[0,:]>zli])
            sigcrit_inv_1 = np.ones((len(zbgrid1[0,zbgrid1[0,:]>zli]),len(zbgrid1[0,zbgrid1[0,:]>zli])))*sigcrit_inv_1

            sigcrit_inv_2 = lensing_kernel(zli,zbgrid2[zbgrid2[:,0]>zli,0])
            sigcrit_inv_2 = np.transpose(np.ones((len(zbgrid2[zbgrid2[:,0]>zli,0]),len(zbgrid2[zbgrid2[:,0]>zli,0])))*sigcrit_inv_2)

            integrand = pz1_integrand[zbgrid1[0,:]>zli] * pz2_integrand[zbgrid2[:,0]>zli] * (R1 * sigcrit_inv_1 + R2 * sigcrit_inv_2)
            
            first_integral = np.trapz(integrand,x = zbgrid2[zbgrid2[:,0]>zli,:len(zbgrid2[zbgrid2[:,0]>zli,0])],axis=0)
            second_integral = np.trapz(first_integral,x=zgrid[zgrid>zli])
            # first_integral = quad(integrand)
            # second_integral = quad(first_integral)
            result[i] = second_integral
        return result
    else:
        zbgrid1,zbgrid2 = np.meshgrid(zgrid,zgrid)
        pz1_integrand = pz(zbgrid1)
        sigcrit_inv_1 = lensing_kernel(zl,zbgrid1)
        pz2_integrand = pz(zbgrid2)
        sigcrit_inv_2 = lensing_kernel(zl,zbgrid2)
        integrand = pz1_integrand * pz2_integrand * (R1 * sigcrit_inv_1 + R2 * sigcrit_inv_2)
        first_integral = np.trapz(integrand,x = zbgrid2,axis=0)
        second_integral = np.trapz(first_integral,x=zgrid)
        return second_integral


z = np.linspace(0,2.5,100)
gg = g_kernel(z)
gb = g_kernel_blend(z)

plt.plot(z,gg,label='single source')
plt.plot(z,gb,label='2 blend sources')
plt.plot(z,pz(z),label='redshift distribution')
plt.xlabel('z')
plt.ylabel('lensing kernel')
plt.legend()
plt.show()