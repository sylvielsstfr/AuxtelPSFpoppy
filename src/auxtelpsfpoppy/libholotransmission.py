# libholotransmission
"""libholotransmission a package to handle interference pattern and transmission of hologram from 2 point sources"""

import numpy as np
import astropy.units as u
from scipy.integrate import quad

# wavelength in m
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

