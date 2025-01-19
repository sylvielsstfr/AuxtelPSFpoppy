# libholotransmission
"""libholotransmission a package to handle interference pattern and transmission of hologram from 2 point sources"""

import numpy as np
import astropy.units as u
from scipy.integrate import quad
import os

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")

#######################
#Auxtel Configuration
#######################

# wavelength
WL = 0.5*u.micron
# pixel scale
# 10 microns pixel , 105 microns per arcsec
PIX_SCALE = 10.0/105.0 # arcsec per pixel
FOV_ARCSEC = 5.0 # must restrict the FOV in focl plane to avoid aliasing
# diameter in m
D = 1.2*u.m
FNUM = 18
# focal length
FL = D*FNUM
# Dccd
DCCD = 180*u.mm
# beam radius at object positon
BEAM_RADIUS =  D/2./FL*DCCD

########################
# Diffraction pattern
########################

NEFF = 150./u.mm  # number of lines per mm at optical center
AH = 1./NEFF      #  line spacing at optical center

#########################
# Hologram Recording
#########################
WLR = 0.639*u.micron  # recording wavelength
dR = 20.0 *u.mm        # distance between sources
DR = dR/WLR/NEFF       # distances between sources plane and emulsion plane to have 

#########################
# Source Beam properties
#########################
C0 = 1.0  #Beam contrast
DPSI0= 0. #Phase difference between the sources 


########################
# Emulsion parameters
########################
GAMMA = 6.5
DELTAMAX = 550*u.nm



def get_data_path():
    """return data path
    """
    return DATA_PATH

def convert_angle_to_0_2pi_interval(angle):
    """Convert the angle in any range into a an angle in 0-2pi range 
    :param angle: angle in radian
    :type angle: float
    :return: angle in range 0-2pi
    :rtype: float
    """
    new_angle = np.arctan2(np.sin(angle), np.cos(angle))
    new_angle = np.where(new_angle < 0,np.abs(new_angle) + 2 * (np.pi - np.abs(new_angle)),new_angle)
    return new_angle

def convert_angle_to_mpi_ppi_interval(angle):
    """
    Convert the angle in range -pi - pi
    :param angle: _description_
    :type angle: _type_
    :return: _description_
    :rtype: _type_
    """
    new_angle = convert_angle_to_0_2pi_interval(angle)
    new_angle = np.where(new_angle > np.pi, new_angle - 2*np.pi,new_angle)
    return new_angle

#################################################
# Diffraction in beam
##################################################

def set_beam_aperture(x,radius = BEAM_RADIUS):
    """Define th transmission through the beam aperture : 1 inside, 0 outside

    :param x: position alonng dispersion axis
    :type x: numpy array with astropy length unit
    :param radius: Beam radius at hologram postion , defaults to BEAM_RADIUS
    :type radius: float with astropy length unit, optional
    :return: array of 0 or 1 depending on the condition
    :rtype: float
    """

    a = np.where(np.logical_or(x<-radius,x>radius),0.,1.)
    return a

####################
# Distances
####################

def D_AM(x,y):
    """Distance in mm between A(-dR/2,0) source and M(x,y) point on emulsion

    :param x: x coordinate of point M in astropy units
    :type x: float with astropy length unit
    :param y: y coordinate of point M in astropy units
    :type y: loat with astropy length unit
    :return: 3D distance between points A(source) and M(point in emulsion)
    :rtype: float with astropy unit
    """
    R2 = DR**2 +(dR/2.)**2+y**2 + dR*x
    return np.sqrt(R2)


def D_BM(x,y):
    """Distance in mm between B(+dR/2,0) source and M(x,y) point on emulsion
    :param x: x coordinate of point M in astropy units
    :type x: float with astropy length unit
    :param y: y coordinate of point M in astropy units
    :type y: float with astropy length unit
    :return: 3D distance between points B(source) and M(point in emulsion)
    :rtypype : float with astropy unit
    """
    
    R2 = DR**2 +(dR/2.)**2+y**2 - dR*x
    return np.sqrt(R2)

##################################
# Interference pattern
##################################

def InterferenceModule1(x,y,wlr=WLR,c=C0,dpsi=DPSI0):
    """True inteferene pattern without any approximation on distances
    :param x: x coordinate of point M 
    :type x: float in astropy lenght unit
    :param y: y coordinate of point M 
    :type y: float in astropy lenght unit
    :param wlr: recording wavelength , defaults to the constant WLR
    :type wlr: float in astropy length unit
    :param c : beamn constrast with default C0
    :type c: float between 0 - 1  
    :param dpsi: phase difference between illuminating sources, defaults to DPSI0
    :type dpsi: float, optional
    :return: the squared module of the interference pattern
    :rtype: float
    """

    #True inteferene pattern
    # compute real positive amplitude from contrast
    
    cos_beta = np.sqrt((np.sqrt(1-c**2)+1)/2.)   
    sin_beta = np.sqrt((1-np.sqrt(1-c**2))/2.)    
    kR = 2.*np.pi/wlr
    RAM = D_AM(x,y)
    RBM = D_BM(x,y)
    
    Ua = cos_beta*np.exp(1j*dpsi/2)*np.exp(1j*kR*RAM)/RAM*DR
    Ub = sin_beta*np.exp(1j*dpsi/2)*np.exp(1j*kR*RBM)/RBM*DR
    
    U = Ua+Ub
    Umod2 = np.abs(U)**2
    return Umod2/2

def InterferenceModule2(x,y,wlr=WLR,c=C0,dpsi=DPSI0):
    """Partially simplified inteference pattern with no position dependence in denominator
    :param x: x coordinate of point M 
    :type x: float in astropy lenght unit
    :param y: y coordinate of point M 
    :type y: float in astropy lenght unit
    :param wlr: recording wavelength , defaults to the constant WLR
    :type wlr: float in astropy length unit
    :param c : beamn constrast with default C0
    :type c: float between 0 - 1  
    :param dpsi: phase difference between illuminating sources, defaults to DPSI0
    :type dpsi: float, optional
    :return: the squared module of the interference pattern
    :rtype: float
    """

    # compute real positive amplitude from contrast
    cos_beta = np.sqrt((np.sqrt(1-c**2)+1)/2.)   
    sin_beta = np.sqrt((1-np.sqrt(1-c**2))/2.)    
    kR = 2.*np.pi/wlr
    RAM = D_AM(x,y)
    RBM = D_BM(x,y)
    
    Ua = cos_beta*np.exp(1j*dpsi/2)*np.exp(1j*kR*RAM)
    Ub = sin_beta*np.exp(1j*dpsi/2)*np.exp(1j*kR*RBM)
    
    U = Ua+Ub
    Umod2 = np.abs(U)**2
    return Umod2/2


def InterferenceModule1D_3(x,wlr=WLR,c=C0,dpsi=0):
    """
    :param x: x coordinate of point M 
    :type x: float in astropy lenght unit
    :param wlr: recording wavelength , defaults to the constant WLR
    :type wlr: float in astropy length unit
    :param c : beamn constrast with default C0
    :type c: float between 0 - 1  
    :param dpsi: phase difference between illuminating sources, defaults to DPSI0
    :type dpsi: float, optional
    :return: the squared module of the interference pattern
    :rtype: float
    """
    cos_beta = np.sqrt((np.sqrt(1-c**2)+1)/2.)
    sin_beta = np.sqrt((1-np.sqrt(1-c**2))/2.)
    kR = 2.*np.pi/wlr
    I = (1.+c*np.cos( (kR*dR*x/DR).to_value() -dpsi))/2.
    return I

def InterferenceModule1D_1(x,wlr=WLR,c=C0,dpsi=DPSI0):
    return InterferenceModule1(x,0.,wlr=wlr,c=c,dpsi=dpsi)
def InterferenceModule1D_2(x,wlr=WLR,c=C0,dpsi=DPSI0):
    return InterferenceModule2(x,0.,wlr=wlr,c=c,dpsi=dpsi)

def InterferenceModule1D_1_A(x,wlr=WLR,c=C0,dpsi=DPSI0):
    return InterferenceModule1D_1((x+dR/2),wlr=WLR,c=c,dpsi=dpsi)
def InterferenceModule1D_2_A(x,wlr=WLR,c=C0,dpsi=DPSI0):
    return InterferenceModule1D_2((x+dR/2),wlr=wlr,c=c,dpsi=dpsi)
def InterferenceModule1D_3_A(x,wlr=WLR,c=C0,dpsi=DPSI0):
    return InterferenceModule1D_3((x+dR/2),wlr=wlr,c=c,dpsi=dpsi) 


def delta_erec(erec,deltamax,thegamma=GAMMA):
    """Effective Optical path length
    :param erec: Fraaction of energy deposit from 0 to a number say 5, 10
    :type erec: float
    :param deltamax: optical depth (depending on wavelength)
    :type deltamax:float
    :param thegamma: _description_, defaults to GAMMA
    :type thegamma: float, optional
    """
    delta_erec =  deltamax/(1+np.exp(-thegamma*(erec-1)))
    return delta_erec


##################################
# Holo Phase
##################################
def Holo_Phase_1A(x,wl,deltamax,wlr=WLR,c=C0,dpsi=DPSI0):
    return 2.*np.pi/wl*delta_erec(InterferenceModule1D_1_A(x,wlr,c,dpsi),deltamax)
def Holo_Phase_2A(x,wl,deltamax,wlr=WLR,c=C0,dpsi=DPSI0):
    return 2.*np.pi/wl*delta_erec(InterferenceModule1D_2_A(x,wlr,c,dpsi),deltamax)
def Holo_Phase_3A(x,wl,deltamax,wlr=WLR,c=C0,dpsi=DPSI0):
    return 2.*np.pi/wl*delta_erec(InterferenceModule1D_3_A(x,wlr,c,dpsi),deltamax)


##################################
# Holo Transmission
##################################

def holo_transmission_1A(x,wl,deltamax,wlr=WLR,c=C0,dpsi=DPSI0):
    y = np.exp(1j*Holo_Phase_1A(x,wl,deltamax,wlr,c,dpsi))
    ycut = np.where(np.logical_or(x<-BEAM_RADIUS,x>BEAM_RADIUS),0.,y)
    return ycut

def holo_transmission_2A(x,wl,deltamax,wlr=WLR,c=C0,dpsi=DPSI0):
    y = np.exp(1j*Holo_Phase_2A(x,wl,deltamax,wlr,c,dpsi)) 
    ycut = np.where(np.logical_or(x<-BEAM_RADIUS,x>BEAM_RADIUS),0.,y)
    return ycut
 
def holo_transmission_3A(x,wl,deltamax,wlr=WLR,c=C0,dpsi=DPSI0):
    y = np.exp(1j*Holo_Phase_3A(x,wl,deltamax,wlr,c,dpsi)) 
    ycut = np.where(np.logical_or(x<-BEAM_RADIUS,x>BEAM_RADIUS),0.,y)
    return ycut



#####################################
# Deltamax vs lambda
# From digitized plot here : https://plotdigitizer.com/app
######################################

def indexmodulationvslambda(wl,Z = np.array([ 1.11813105e+09, -3.80887867e+06,  1.02558810e+04, -4.05052174e+00,
        9.89018007e-02])):
    """return deltamax vs lambda

    :param wl: wavelength in astropy length units
    :type wl: float or array
    """
    #Z = np.array([ 1.11813105e+09, -3.80887867e+06,  1.02558810e+04, -4.05052174e+00,
    #    9.89018007e-02])

    indexmod  = Z[4] + Z[3]*u.nm/wl + Z[2]*(u.nm)**2/wl**2 + Z[1]*(u.nm)**3/wl**3 + Z[0]*(u.nm)**4/wl**4
    return indexmod




def deltamaxvslambda(wl,Z = np.array([ 9.50123454e+12, -4.39834684e+10,  1.10751635e+08, -8.05697911e+04,
        5.16109943e+02])):
    """_summary_

    :param wl: wavelength in astropy length units
    :type wl: float or array of float
    :param Z: coefficients from polynomial fit, defaults to array([ 9.50123454e+12, -4.39834684e+10,  1.10751635e+08, -8.05697911e+04, 5.16109943e+02]
    :type Z: array of floats, optional
    """

    deltamax  = Z[4] + Z[3]*u.nm/wl + Z[2]*(u.nm)**2/wl**2 + Z[1]*(u.nm)**3/wl**3 + Z[0]*(u.nm)**4/wl**4
    return  deltamax*u.nm


##################################
# FFT
##################################
def ComputeFFT(r,dxe):
    """Compute the DFT
     
    :param r: transmission array
    :type r: 1D complex array type
    :param dxe: x- sampling (prefer in mm)
    :type dxe: float
    :return: FFT results in term of frequency, real, imag, module and phase
    :rtype: freq,real,imag,module,phase _: 5 real arrays
    """
    
    sp = np.fft.fft(r)
    # the frequency in units of mm^-1
    # do a deep copy
    freq = np.array(np.fft.fftfreq(r.shape[-1], d=dxe))
    real = np.array(sp.real)
    imag = np.array(sp.imag)
    module = np.array(np.abs(sp))
    phase = np.array(np.angle(sp))
    
    freq = np.fft.fftshift(freq)
    real = np.fft.fftshift(real)
    imag = np.fft.fftshift(imag)
    module = np.fft.fftshift(module)
    phase = np.fft.fftshift(phase)
    return freq,real,imag,module,phase
