import numpy as np
from dielectric import DielectricConstant
from os import path
from scipy import constants
import MDAnalysis as mda

def water_dielectric(dir):
    top = path.join(dir, 'step5.tpr')
    trj = path.join(dir, 'md.xtc')
    u = mda.Universe(top, trj)
    diel = DielectricConstant(u.atoms, temperature=303.15, make_whole=False, start=5000)
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
    data = np.genfromtxt(visco, comments='@', skip_header=18)
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

def rdf_diff(dir):
    rdf = np.load(path.join(dir, 'rdf.npy'))
    return rdf

def check_sim(dir):
    gro = path.join(dir, 'step5.gro')
    return path.isfile(gro)

def self_diff(dir):
    data = path.join(dir, "msd.xvg")
    with open(data, 'r') as f:
        for i, line in enumerate(f):
            if i == 19:
                file_content = line.strip()
    return float(file_content.split()[4])

def objective(dir, ref):
    comp = check_sim(dir)
    if comp:
        comp = 1
        diel = water_dielectric(dir)
        dens = water_density(dir)
        visc = shear_viscosity_einstein(dir)
        diff = self_diff(dir)
        rdf = rdf_diff(dir)
        obj = np.sqrt(np.sum(((np.array([diel, dens, visc*1000, diff]) - ref[0])/ref[0])**2) + (np.sum(np.abs(rdf-ref[1]))/37)**2)
    else:
        comp = 0
        diel = 0
        dens = 0
        visc = 0
        diff = 0
        obj = 0
    return np.array([comp, diel, dens, visc, diff, obj])