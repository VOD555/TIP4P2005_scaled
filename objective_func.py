import numpy as np
from dielectric import DielectricConstant
from os import path
from scipy import constants, optimize
import MDAnalysis as mda
import matplotlib.pyplot as plt

def water_dielectric(dir):
    diels = []
    for i in range(5):
        top = path.join(dir, 'md{}/nvt.tpr'.format(i))
        trj = path.join(dir, 'md{}/nvtc.xtc'.format(i))
        u = mda.Universe(top, trj)
        diel = DielectricConstant(u.atoms, temperature=303.15, make_whole=False, start=5000)
        diel.run()
        diels.append(diel.results['eps_mean'])
    return np.mean(diels)

def shear_viscosity_einstein(dir):
    '''
    Read the output file from gromacs and calcualte average viscosity.
    visco:
        xvg file containing the viscosity.
    return the average viscosity
    '''
    # # Define a fitting function.
    # def curve(x, A, a, t1, t2):
    #     return A*a*t1*(1-np.exp(-x/t1))+A*(1-a)*t2*(1-np.exp(-x/t2))
    
    # Load the file into a numpy array
    viscosity = []
    for i in range(5):
        visco = path.join(dir, 'md{}/visco.xvg'.format(i))
        data = np.genfromtxt(visco, comments='@', skip_header=18)
        viscosity.append(data[:, 1])
    viscosity = np.array(viscosity)
    ydata = np.mean(viscosity, axis=0)[500:1000]
    # xdata=data[:1000, 0]/1000
    # ydata = np.mean(viscosity, axis=0)[:1000]
    # popt, pcov = optimize.curve_fit(curve, xdata, ydata)
    # y = curve(xdata, *popt)
    # plt.plot(xdata, ydata.T)
    # plt.plot(xdata,y)
    # plt.xlabel('time (ns)')
    # plt.ylabel('viscosity (cp)')
    # plt.savefig(path.join(dir, 'viscosity.png'))
    # plt.close()

    # Calculate the viscosity at infinite time limit.
    # final_vis = popt[0]*popt[1]*popt[2]+popt[0]*(1-popt[1])*popt[3]
    return np.mean(ydata)

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
    diffu = []
    for i in range(5):
        data = path.join(dir, "md{}/msd.xvg".format(i))
        with open(data, 'r') as f:
            for i, line in enumerate(f):
                if i == 19:
                    file_content = line.strip()
        diffu.append(float(file_content.split()[4]))
    return np.mean(diffu)

def objective(dir, ref):
    comp = check_sim(dir)
    if comp:
        comp = 1
        diel = water_dielectric(dir)
        dens = water_density(dir)
        visc = shear_viscosity_einstein(dir)
        diff = self_diff(dir)
        rdf = rdf_diff(dir)
        obj = np.sqrt((np.sum(((np.array([diel, dens, visc, diff]) - ref[0])/ref[0])**2*np.array([4,4,1,1])) + (np.sum(np.abs(rdf-ref[1]))/37)**2)/11)
    else:
        comp = 0
        diel = 0
        dens = 0
        visc = 0
        diff = 0
        obj = 0
    return np.array([comp, diel, dens, visc, diff, obj])