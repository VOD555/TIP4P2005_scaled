import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rdf

if __name__ == '__main__':
    top = 'step5.tpr'
    trj = 'md.xtc'
    u = mda.Universe(top, trj)
    oxygen = u.select_atoms('name OW1')
    o_o = rdf.InterRDF(oxygen, oxygen, 307, (1.005, 10.215))
    o_o.run()
    np.save('rdf', o_o.results.rdf)
