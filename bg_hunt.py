import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy.special import factorial as fact

plt.rcParams.update({'font.size': 15})
plt.rc('axes', labelsize=20) 
fig, ax = plt.subplots()

#fig, (axList) = plt.subplots(1, 2, figsize=(10, 10))
#axList = axList.flatten()
#fig.tight_layout(pad=4.0)

### paramters (SI units)

dim = 20 # dimension of angular momentum space
mid=int((4*dim-1)/2)-1
hbar = 1.05*10**(-34) # planck's constant (J*s)
e = 1.602*10**(-19) # electron charge (C)
J2meV = 10**3/e # joules to meV conversion factor
a0 = 2.46*10**(-10) # graphene carbon-carbon spacing (m)

eta = 1 # valley
i=complex(0,1)

### paramters

# hopping (J) new
# g0 = 2.61*10**3/J2meV
# g1 = 0.361*10**3/J2meV
# g3 = 0
# g4 = 0.138*10**3/J2meV

# old
g0 = 2.61*10**3/J2meV
g1 = 0.361*10**3/J2meV
g3 = 0#-0.283*10**3/J2meV
g4 = -0.138*10**3/J2meV

# 
Dp = 0.015*10**3/J2meV

# symmetry-breaking
D10=9.7/J2meV
EZ=3.58/J2meV

# B=0 (J)
v0 = np.sqrt(3)*a0*np.abs(g0)/(2*hbar)

# standard operators
def a_creat():
    # return the matrix form of the creation operator
    X = np.zeros((dim,dim))
    for i in range(dim-1):
        X[i+1,i] = np.sqrt(i+1)
    
    return np.matrix(X)

def a_annih():
    # return the matrix form of the annihilation operator
    return np.transpose(a_creat())

### initialize operators
id_d=np.identity(dim)
z=np.zeros((dim,dim))
a=a_annih()
ad=a_creat()

###

def bg_z(k,u,N):
    # bilayer graphene in zero magnetic field (single particle)
    # u: electric field between layers
    # return eigenstate of Nth level, and all energy levels
    # eta=+-1 for K,K' respectively

    pi=hbar*(k[0]+i*k[1])
    pid=hbar*(k[0]-i*k[1])

    h=J2meV*np.array([[eta*u/2,0,v0*pid,-v4*pid],
                        [0,-eta*u/2,-v4*pi,v0*pi],
                        [v0*pi,-v4*pid,(eta*u/2+Dp),g1],
                        [-v4*pi,v0*pid,g1,(-eta*u/2+Dp)]])
    #if (eta==-1):
    #    basis_change=np.block([[z,id_d,z,z],[id_d,z,z,z],[z,z,z,id_d],[z,z,id_d,z]])
    #    h=basis_change@h@basis_change

    eigenvalue, eigenvector=np.linalg.eig(h)
    eigenvector=np.transpose(eigenvector)
    eig_vecs_sorted=eigenvector[eigenvalue.argsort(),:]
    
    return np.sort(eigenvalue), eig_vecs_sorted[N]

def bg_h(B,u,N):
    # bilayer graphene in magnetic field (single particle)
    # u: electric field between layers
    # return eigenstate of Nth level, and all energy levels
    # eta=+-1 for K,K' respectively
    # eta=1 basis: A,B',B,A', eta=-1 basis: B',A,A',B

    # hw0=30.6*np.sqrt(B)/J2meV # J
    
    
    l=np.sqrt(hbar/(e*B))
    hw0=hbar*v0*np.sqrt(2)/l
    
    # omegas
    w0=hw0
    w4=g4/g0*w0
    w3=g3/g0*w0
    
    # # new
    # h=J2meV*hw0*np.block([[eta*u/(2*hw0)*id_d,z,ad,-g4/g0*ad],
                          # [z,-eta*u/(2*hw0)*id_d,-g4/g0*a,a],
                          # [a,-g4/g0*ad,(eta*u/2+Dp)/hw0*id_d,g1/hw0*id_d],
                          # [-g4/g0*a,ad,g1/hw0*id_d,(-eta*u/2+Dp)/hw0*id_d]])

    # old
    h=J2meV*np.block([[eta*u/2*id_d,w3*a,w4*ad,w0*ad],
                     [w3*ad,-eta*u/2*id_d,w0*a,w4*a],
                     [w4*a,w0*ad,(-eta*u/2+Dp)*id_d,g1*id_d],
                     [w0*a,w4*ad,g1*id_d,(eta*u/2+Dp)*id_d]])

    if (eta==-1):
        basis_change=np.block([[z,id_d,z,z],[id_d,z,z,z],[z,z,z,id_d],[z,z,id_d,z]])
        h=basis_change@h@basis_change

    eigenvalue, eigenvector=np.linalg.eig(h)
    
    eigenvector=np.transpose(eigenvector)
    eig_vecs_sorted=eigenvector[eigenvalue.argsort(),:]
    eig_vals_sorted=np.sort(eigenvalue)
    
    # remove high-level states (there is no maximum state)
    ad_diag=np.array(scipy.linalg.block_diag(ad,ad,ad,ad))
    k=0
    while (k<len(eig_vecs_sorted)):
        raised_eig=ad_diag@np.transpose(eig_vecs_sorted[k])
        if (not np.any(raised_eig)):
            eig_vecs_sorted=np.delete(eig_vecs_sorted,k,0)
            eig_vals_sorted=np.delete(eig_vals_sorted,k,0)
            k+=-1
        k+=1

    # return eigenvalue, eig_vecs_sorted[N]
    return eig_vals_sorted, np.array(eig_vecs_sorted)

def sublattice_polarization(a, print_sublattice=0):
    # sublattice polarization alpha
    A=np.max(np.abs(a[0:dim]))
    Bp=np.max(np.abs(a[dim:2*dim]))
    
    # old interchange Ap, B
    Ap=np.max(np.abs(a[2*dim:3*dim]))
    B=np.max(np.abs(a[3*dim:]))
    
    # print(np.abs(Ap)/np.abs(A))
    
    if (print_sublattice):
        print("A:",A,"\nBp",Bp,"\nB",B,"\nAp",Ap)
    
    alpha=np.around(A**2-Bp**2+B**2-Ap**2,3)
    return alpha
    # print("alpha", np.around(A**2-Bp**2+B**2-Ap**2,3))

def sym_breaking(u):
    # u: interlayer electric potential energy difference

    # symmetry-breaking of ZLL, 8 flavors
    spin=[1/2,-1/2]
    alpha=[1,0.63] 
    N=[0,1]
    eta=[-1,1]
    
    e_symbreak=np.zeros((2,2,2))
    for i in range(2): # spin
        for j in range(2): # orbital
            for k in range(2): # valley
                e_symbreak[i,j,k]=-EZ*spin[i]+N[j]*D10-u/2*eta[k]*alpha[j]
    
    return e_symbreak

###

def band_structure(plot=0):
    npts=100
    
    # theory
    #energy = np.zeros((npts,2*dim))
    
    # B\neq 0
    energy = np.zeros((npts,4*dim))
    #B=np.linspace(10**(-3),35,npts) # (T)
    u_range=10/J2meV
    u=np.linspace(-u_range,u_range,npts)
    
    # B=0
    # energy=np.zeros((npts,4))
    # k_range=1/(2*np.pi*a0)
    # kx=np.linspace(-k_range,k_range,npts)
    # ky=np.linspace(-a0*10,a0*10,npts)
    
    for i in range(npts):
    
        # B\neq 0
        # energy[i]=bg_h(B[i],0,0)[0]
        
        # B=31
        energy[i]=bg_h(31,u[i],0)[0]
        
        # B=0
        #k=[kx[i],0]
        #energy[i]=bg_z(k,0,0)[0]
        
        # theory
        #energy[i]=bg_e(B[i],0)
        
    
    if (plot==1):
        # plt.plot(B,energy,color="red")
        # axList[0].plot(B,energy,color="red")
        
        # B=31
        plt.plot(u*J2meV,energy[2*dim,2*dim+2])
        plt.ylim([-20,20])
        
        print(energy[2*dim:2*dim+2])
        
        # B=0
        # plt.plot(kx,energy)
        
        
        # plt.title("Bilayer graphene energy levels")
        # plt.xlabel("Magnetic field (T)")
        # plt.ylabel("Energy (meV)")
        plt.show()
    
    return energy

evals, evecs = bg_h(31,0/J2meV,0)
# print("energy", evals)

r=3
for k in range(mid-r,mid+r+1):
    print("energy", evals[k])
    print("polarization", sublattice_polarization(evecs[k]))
    print("--")

###

# npts=100
# u_range=10/J2meV
# u=np.linspace(-u_range,u_range,npts)
# energy=np.zeros((npts,8))
# for i in range(len(u)):
    # energy[i]=np.ndarray.flatten(sym_breaking(u[i]))*J2meV

# plt.plot(u,energy)

###

# E = band_structure(1)

# # DOS
# import pandas as pd
# EinBZ = E.flatten()
# dataframe = pd.DataFrame(EinBZ)
# dataframe.to_csv('EinBZ.csv', index=False, sep=',')
# import seaborn as sns
# data = np.loadtxt('EinBZ.csv', delimiter=',')

# # we want DOS at fixed B, over k
# maxE = 10**3
# # sns.displot([i for i in data if (i<maxE and i>-maxE)], kind="hist", kde=False, bins=10**2)

#ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
#plt.xlabel('Energy (meV)')
#plt.ylabel('DOS')
#plt.title('Density of states')
#plt.show()