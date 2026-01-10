import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import scipy.linalg as sla

# CORE FUNCTIONS for transforming trajectories to contact map graph networks

def kabsch_align(P, Q):
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    U, _, Vt = sla.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    return (Pc @ R) + Q.mean(axis=0)

def align_traj(coords, ref=None):
    T, N, _ = coords.shape
    if ref is None:
        ref = coords[0]
    aligned = np.empty_like(coords)
    for t in range(T):
        aligned[t] = kabsch_align(coords[t], ref)
    return aligned

def get_contact_maps(u=None, top=None, traj=None):
    '''
    Returns contact_maps matrx of shape (F,N,N) where F is frame count and 
    N is number of residues
    '''
    if u == None:
        u = mda.Universe(top, traj)

    atoms = u.select_atoms("name CA")   # or any atom selection you want
    n = atoms.n_atoms
    cutoff = 8.0  # Å

    contact_maps = np.empty((len(u.trajectory), n, n), dtype=np.int8)

    for i, ts in enumerate(u.trajectory):
        # positions for this frame
        pos = atoms.positions

        # full NxN distance matrix (symmetric)
        D = distances.distance_array(pos, pos)

        # boolean contact matrix, zero diagonal (no self-contact)
        C = (D < cutoff)
        np.fill_diagonal(C, False)

        contact_maps[i] = C.astype(np.int8)

    return contact_maps

def flatten_upper_bits(maps):
    ''''''
    U, N, N2 = maps.shape
    assert N == N2
    tri = np.triu_indices(N, k=1)
    Mbits = len(tri[0])
    flat = maps[:, tri[0], tri[1]].astype(np.uint8)  # (U, Mbits)
    return flat, tri, Mbits

def maps_to_upper_ints(maps):
    '''
    Convert bool contact maps to upper traingle 
    '''
    F, N, N2 = maps.shape # U = number of maps
    assert N == N2
    
    flat, tri, Mbits = flatten_upper_bits(maps) # flatten to bit vector shape (U, Mbits)
    # this transforms each contact map's upper triangle to a binary vector

    # this will contain a single integer representation of each unique contact map
    ints = np.empty(F, dtype=object)
    for idx in range(F):
        row = flat[idx] # get flattened triangle for idx'th map

        val = 0
        # here is the actual bit vector encoding
        for b in reversed(row):
            val = (val << 1) | int(b)

        ints[idx] = val 

    return ints, Mbits, tri # Mbits and triused for backmapping int to binary 

def get_unique_maps(contact_maps):
    '''
    Creates mapping for each frame in trajectory to a unique contact map id
    and provides the index of each unique maps first occurence in trajectory.
    '''
    F, N, _ = contact_maps.shape
    assert N == contact_maps.shape[2]

    ints, Mbits, tri = maps_to_upper_ints(contact_maps)  # ints: length-F array of Python ints

    first_idx_of = {}
    frame_to_uid = np.empty(F, dtype=int)
    first_inds = []
    for f, v in enumerate(ints):
        if v in first_idx_of:
            frame_to_uid[f] = first_idx_of[v]
        else:
            uid = len(first_inds)
            first_idx_of[v] = uid
            frame_to_uid[f] = uid
            first_inds.append(f)

    inds = np.array(first_inds, dtype=int)
    unique_maps = contact_maps[inds]

    print("Frames:", F, "N:", N, "Unique maps:", len(inds))
    return unique_maps, frame_to_uid, inds