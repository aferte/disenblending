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

def delta_p(z,zmax=3,ngrid=100,eps = 1.005):
    '''integral of the lensing kernel '''
    
    '''z to ingrate on'''    
    zgrid = np.linspace(0,zmax,ngrid)
    pzgrid = pz(zgrid)
    if len(z) > 1:
        result = np.zeros_like(z)
        for i,thisz in enumerate(z):
            integrand = pzgrid[zgrid>(thisz*eps)]/lensing_kernel(thisz,zgrid[zgrid>(thisz*eps)])
            result[i] = np.trapz(integrand,x=zgrid[zgrid>(thisz*eps)])
        return result
    else:
        integrand = pzgrid[zgrid>z]/lensing_kernel(z,zgrid[zgrid>z])
        result = np.trapz(integrand,zx=zgrid[zgrid>z])
    return result


def main(argv):
    z = np.linspace(0,2.5,100)
    dp = delta_p(z,ngrid=10001)

    plt.plot(z,dp,label='effective delta p(z)')
    plt.plot(z,pz(z),label='p(z)')
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.legend()
    #plt.yscale('log')    
    #plt.show()
    plt.savefig('lensing_kernel_modification.png')
    pdb.set_trace()
    
if __name__ == "__main__":
    import pdb, traceback, sys
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
