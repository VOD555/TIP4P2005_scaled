import numpy as np
from dielectric import DielectricConstant
from os import path
from scipy import constants
import MDAnalysis as mda

def water_dielectric(dir):
    top = path.join(dir, 'step5.tpr')
    trj = path.join(dir, 'md.xtc')
    u = mda.Universe(top, trj)
    diel = DielectricConstant(u.atoms, temperature=298, make_whole=False, start=5000)
    diel.run()
    return diel.results['eps_mean']

def shear_viscosity_einstein(dir):
    '''
    Read the output file from gromacs and calcualte average viscosity.
    visco:
        xvg file containing the viscosity.
    return the average viscosity
    '''
    # Load the file into a numpy array
    visco = path.join(dir, 'evisco.xvg')
    data = np.genfromtxt(visco, comments='@', skip_header=16)
    return np.mean(data[:,4])

def water_density(dir):
    top = path.join(dir, 'step5.tpr')
    trj = path.join(dir, 'md.xtc')
    u = mda.Universe(top, trj)
    volume = 0
    for ts in u.trajectory:
        volume += ts.volume

    volume = volume/u.trajectory.n_frames
    density = u.atoms.total_mass()/volume*constants.atomic_mass*10**27
    return density

def check_sim(dir):
    gro = path.join(dir, 'step5.gro')
    return path.isfile(gro)

def objective(dir, ref):
    comp = check_sim(dir)
    if comp:
        comp = 1
        diel = water_dielectric(dir)
        dens = water_density(dir)
        visc = shear_viscosity_einstein(dir)
        obj = np.sqrt(np.sum((np.array([diel, dens, visc]) - ref)**2)/3)
    else:
        comp = 0
        diel = 0
        dens = 0
        visc = 0
        obj = 0
    return np.array([comp, diel, dens, visc, obj])