import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from astropy import cosmology
from astropy import constants as const

H0,Om = 68.3, 0.299
cosmo = cosmology.FlatLambdaCDM(H0,Om) 

'''create p(z)'''
def redshift_distrib(z): 
    z0 = 0.5
    alpha = 1.
    beta = 2.
    redshift_distrib = z**alpha *np.exp(-(z/z0)**beta)
    return redshift_distrib


def lensing_kernel(z1,z2):
    factor = const.c**2 / (4*np.pi*const.G)
    sig_crit = factor * cosmo.angular_diameter_distance(z2) / cosmo.angular_diameter_distance(z1) /  cosmo.angular_diameter_distance_z1z2(z1,z2)
    return 1/sig_crit.cgs.value

def int_lensing_kernel(z,z_source,z_start,z_end):
    int_lensing_kernel = quad(lensing_kernel,z_start,z_end,args=())
    return int_lensing_kernel

def kernel_total(z,z1,z2):
    kernel_total = redshift_distrib(z)*redshift_distrib(z)*[R1*lensing_kernel(z,z1) * R2*rlensing_kernel(z,z2)]*cosmo.e_func()/H0
    return kernel_total


'''responsivity for blended galaxies'''
R1 = 0.7
R2 = 0.3
z_start = 0.001
z = np.linspace(0,3,100)

g_blend_total = []
for i in range(len(z)):
    g_blend_total[i] = quad ( lambda z_source2: redshift_distrib(z)*quad( lambda z_source1: redshift_distrib(z)* (R1*lensing_kernel(z,z_source1)  + R2*lensing_kernel(z,z_source2))  ,z_start,10 ),z_start,10)



'''plot redshift distribution'''
plt.plot(z,redshift_distrib(z,0.5,1,2))
plt.show()

plt.plot(z,lensing_kernel(0.5,z))
plt.show()


# plt.plot(z,kernel_total(z,0.5,1))
# plt.show()






