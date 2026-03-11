import numpy as np
from typing import Tuple

def read_cco(ifile: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Read data from CCO, taking advantage of fixed file setup

    Returns:
    --------
        d : int
            system dimensionality
        num_particles : int
            number of particles
        lattice_vectors : np.ndarray
            primitive lattice vectors of 3D system
        coordinates : np.ndarray
            positions of all individual spheres
    """
    lattice_vectors = []
    coordinates = []

    with open(ifile, "r", encoding="utf-8") as f:
        lines = f.readlines()
        d = int(lines[0].strip("\n"))

        # d counts number of vectors, -1 for dimension in the first line
        n_particles = len(lines) - (d + 1)

        for i in range(1, d + 1):  # Get all basis vectors
            basis_vector = lines[i].strip("\n").split()
            lattice_vectors.append([float(a) for a in basis_vector[:d]])

        for j in range(d + 1, len(lines)):  # Get all coordinates
            coord = lines[j].strip("\n").split()
            coordinates.append([float(a) for a in coord[:d]])

    return d, n_particles, np.array(lattice_vectors), np.array(coordinates)