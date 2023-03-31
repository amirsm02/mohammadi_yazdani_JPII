import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy.special import factorial as fact

plt.rcParams.update({'font.size': 15})
plt.rc('axes', labelsize=20) 
fig, ax = plt.subplots()

### paramters (SI units)

dim = 50
hbar = 1.05*10**(-34) # J*s
e = 1.602*10**(-19) # Coulomb
J2meV = 10**3/e
# a = 2.46*10**(-10) # m
eta = 1

# J
g0 = 2.61*10**3/J2meV
g1 = 0.361*10**3/J2meV
g4 = 0.138*10**3/J2meV
Dp = 0.015*10**3/J2meV


def a_creat():
    # return the matrix form of the creation operator
    X = np.zeros((dim,dim))
    for i in range(dim-1):
        X[i+1,i] = np.sqrt(i+1)
    
    return np.matrix(X)

def a_annih():
    # return the matrix form of the annihilation operator
    return np.transpose(a_creat())

def bg_h(B,u,N):
    # bilayer graphene in magnetic field (single particle)
    # u: electric field between layers
    # return eigenstate of Nth level and all energy levels
    # eta=+-1 for K,K' respectively

    hw0=30.6*np.sqrt(B)/J2meV # J
    
    id_d=np.identity(dim)
    z=np.zeros((dim,dim))
    a=a_annih()
    ad=a_creat()
    
    h=J2meV*hw0*np.block([[eta*u/(2*hw0)*id_d,z,ad,-g4/g0*ad],
                            [z,-eta*u/(2*hw0)*id_d,-g4/g0*a,a],
                            [a,-g4/g0*ad,(eta*u/2+Dp)/hw0*id_d,g1/hw0*id_d],
                            [-g4/g0*a,ad,g1/hw0*id_d,(-eta*u/2+Dp)/hw0*id_d]])

    if (eta==-1):
        basis_change=np.block([[z,id_d,z,z],[id_d,z,z,z],[z,z,z,id_d],[z,z,id_d,z]])
        h=basis_change@h@basis_change

    eigenvalue, eigenvector=np.linalg.eig(h)
    eigenvector=np.transpose(eigenvector)
    eig_vecs_sorted=eigenvector[eigenvalue.argsort(),:]
    
    return np.sort(eigenvalue), eig_vecs_sorted[N]

z=int(4*dim/2)-1

def sublattice_polarization(eigvec, print_sublattice=0):
    A=np.max(np.abs(eigvec[0,0:dim]))
    Bp=np.max(np.abs(eigvec[0,dim:2*dim]))
    B=np.max(np.abs(eigvec[0,2*dim:3*dim]))
    Ap=np.max(np.abs(eigvec[0,3*dim:]))
    # print("A:",A,"\nBp",Bp,"\nB",B,"\nAp",Ap)
    print("N=", i, "alpha", np.around(A**2-Bp**2+B**2-Ap**2,3), "energy", np.around(energy_h1[z+i],2))
    

for i in range(-5,5):
    energy_h1, eigvec = bg_h(31,0,z+i) # meV
    sublattice_polarization(eigvec)

energy_h1, eigvec = bg_h(31,0,z+1)
eigvec=np.around(eigvec,3)
print("A", eigvec[0,0:dim])
print("Bp", eigvec[0,dim:2*dim])
print("B", eigvec[0,2*dim:3*dim])
print("Ap", eigvec[0,3*dim:])

norm=0
for i in range(4*dim):
    norm+=eigvec[0,i]**2
print("norm:", norm)