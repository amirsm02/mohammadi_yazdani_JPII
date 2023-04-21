import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy.special import factorial as fact
import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 15})
plt.rc('axes', labelsize=20) 
fig, ax = plt.subplots()

### paramters (SI units)

dim = 20 # dimension of angular momentum space
mid=int((4*dim-1)/2)-1
pt_den = 300 # density of BZ grid points

hbar = 1.05*10**(-34) # planck's constant (J*s)
e = 1.602*10**(-19) # electron charge (C)
J2meV = 10**3/e # joules to meV conversion factor
a0 = 2.46*10**(-10) # graphene AA distance

# eta = 1 # valley
i=complex(0,1)

### paramters

# hopping (J)
g0 = 2.61*10**3/J2meV
g1 = 0.361*10**3/J2meV
g3 = 0 #-0.283*10**3/J2meV #0 
g4 = -0.138*10**3/J2meV # on-site energy due to hBN

# 
Dp = 0.015*10**3/J2meV

# symmetry-breaking
D10=9.7/J2meV # 
EZ=3.58/J2meV # Zeeman splitting

# B=0 (J)
v0 = np.sqrt(3)*a0*np.abs(g0)/(2*hbar)
v4 = np.sqrt(3)*a0*np.abs(g4)/(2*hbar)

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

def hex_grid(r, psi):
    R = 2/np.sqrt(3)*r
    total_pts = int(3*pt_den**2)
    k = [[],[]]
    step = r/pt_den
    
    # rectangle
    y = step/2
    i = 0
    while (y <= R/2):
        n = int(np.ceil(r/step))
        k[1] = np.append(k[1], np.array([y]*n))
        k[0] = np.append(k[0], np.arange(step/2,r,step))
        i = i + pt_den
        y = y + r / pt_den
        
    # triangle
    x = r
    i = 0
    while (y <= R):
        n = int(np.ceil((x-step/2)/step))
        k[1] = np.append(k[1], np.array([y]*n))
        k[0] = np.append(k[0], np.arange(step/2,x,step))
        i = i + n
        y = y + r / pt_den
        x = x - r / pt_den * np.sqrt(3)
    
    k[1] = np.append(k[1], -k[1])
    k[0] = np.append(k[0], k[0])
    
    k[1] = np.append(k[1], k[1])
    k[0] = np.append(k[0], -k[0])
    
    sites = np.zeros((len(k[0]),2))

    M_psi = np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]])
    for i in range(len(k[0])):
        sites[i] = M_psi @ np.array([k[0][i],k[1][i]])
        
    return sites
    
# sites=hex_grid(1,0)
# plt.scatter(sites[:,0],sites[:,1],s=2)
# plt.show()

def f(k):
    s=a0/(2*np.cos(np.pi/6))
    d1=s*np.array([0,1])
    d2=s*np.array([-np.sqrt(3)/2,-1/2])
    d3=s*np.array([np.sqrt(3)/2,-1/2])
    return np.exp(-i*np.dot(d1,k))+np.exp(-i*np.dot(d2,k))+np.exp(-i*np.dot(d3,k))

###

def bg_z(eta,k,u):
    # low energy
    # bilayer graphene in zero magnetic field (single particle)
    # u: electric field between layers
    # return eigenstate of nth level, and all energy levels
    # eta=+-1 for K,K' respectively

    # low energy
    #pi=hbar*(k[0]-i*k[1])
    #pid=hbar*(k[0]+i*k[1])
    
    # general energy
    pi=f(k)
    pid=np.conjugate(pi)

    h=J2meV*np.array([[eta*u/2,g3*pi,g4*pid,g0*pid],
                      [g3*pid,-eta*u/2,g0*pi,g4*pi],
                      [g4*pi,g0*pi,(-eta*u/2+Dp),g1],
                      [g0*pi,g4*pid,g1,(eta*u/2+Dp)]])
    
    # alpha=[1,0.63] 
    # N=[0,1]
    # eta=[-1,1]
    
    # e_symbreak=np.zeros((2,2,2))
    # for j in range(2): # orbital
        # for k in range(2): # valley
            # e_symbreak[i,j,k]=N[j]*D10-u/2*eta[k]*alpha[j]
    
    # return e_symbreak

    eigenvalue, eigenvector=np.linalg.eig(h)
    eigenvector=np.transpose(eigenvector)
    eig_vecs_sorted=eigenvector[eigenvalue.argsort(),:]
    eig_vals_sorted=np.sort(eigenvalue)
    
    return eig_vals_sorted #, np.array(eig_vecs_sorted)

def bg_h(eta,B,u,N):
    # bilayer graphene in magnetic field (single particle)
    # u: electric field between layers
    # return eigenstate of Nth level, and all energy levels
    # eta=+-1 for K,K' respectively
    # eta=1 basis: A,B',B,A', eta=-1 basis: B',A,A',B
    
    l=np.sqrt(hbar/(e*B))
    hw0=hbar*v0*np.sqrt(2)/l
    
    # omegas
    w0=hw0
    w4=g4/g0*w0
    w3=g3/g0*w0
    
    # print(w0**2/g1*(-g4/g0+Dp/g1)*J2meV)
   
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

    return np.real(eig_vals_sorted), np.array(eig_vecs_sorted)

def sublattice_polarization(a, print_sublattice=0):
    # sublattice polarization alpha
    
    A=np.max(np.abs(a[0:dim]))
    Bp=np.max(np.abs(a[dim:2*dim]))
    Ap=np.max(np.abs(a[2*dim:3*dim]))
    B=np.max(np.abs(a[3*dim:]))
    
    print("A'/A", Ap/A)
    print("B", B)
    
    if (print_sublattice):
        print("A:",A,"\nBp",Bp,"\nB",B,"\nAp",Ap)
    
    alpha=np.around(A**2+B**2-Ap**2-Bp**2,3)
    return alpha

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

def bg_bands():
    
    u=0.06# 0.05 # -10**2 #eV
    
    
    energy=bg_h(1,0,u/(J2meV*10**(-3)),0)[0]
    
    #eta2=bg_h(-1,6,u/(J2meV*10**(-3)),0)[0]
    #energy=np.append(eta1,eta2)
    
    for k in range(len(energy)):
        if (np.abs(energy[k])<50):
            print(energy[k])
    
    #r=3
    #for k in range(mid-r,mid+r+1):
       #print("energy", energy[k])
       # print("polarization", sublattice_polarization(evecs[k]))
       #print("--")
    
    # kr=np.around(4*np.pi/(3*a0)*(np.sqrt(3)/2),3) # BG
    # sites=hex_grid(kr,np.pi/2)
    # energy=np.zeros((len(sites),2,4)) # 4 energy levels, 2 valleys
    # for m in range(len(sites)):
        # k=np.array([sites[m,0],sites[m,1]])
        # energy[m,0]=bg_z(1,k,u/(J2meV*10**(-3)))
        # energy[m,1]=bg_z(-1,k,u/(J2meV*10**(-3)))
        
    
    
    # data=energy[:,1]
    # abs_max = max(np.abs(data.min()), np.abs(data.max()))
    
    # # color plot
    # from matplotlib import colors
    # from matplotlib import cm as cmx

    # cNorm  = colors.Normalize(vmin=-abs_max, vmax=abs_max)
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('seismic'))

    # color_ = [0]*len(sites)
    # for m in range(len(sites)):
        # color_[m] = scalarMap.to_rgba(data[m])
    
    # rng=2/np.sqrt(3)*kr
    # plt.gca().set_aspect(1.0)
    # plt.xlim([-rng,rng])
    # plt.ylim([-rng,rng])
    # plt.scatter(sites[:,0], sites[:,1], s=1, color=color_, marker='s')
    
    # DOS
    # print(np.abs(np.max(energy)))
    # print(energy)
    # data = pd.DataFrame(energy.flatten())
    # print(data)
    
    energy = energy.flatten()
    dataframe = pd.DataFrame(energy)
    dataframe.to_csv('energy.csv', index=False, sep=',')
    data = np.loadtxt('energy.csv', delimiter=',')
    
    maxE = 80 # (meV)

    sns.displot([i for i in data if (i<maxE and i>-maxE)], kind="hist", kde=False, bins=5*maxE)
    plt.xlabel('Energy (meV)')
    plt.ylabel('DOS')
    plt.title('Density of states')
    plt.show()

# bg_bands()

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
        # energy[i]=bg_h(31,u[i],0)[0]
        
        # B=0
        k=[kx[i],0]
        energy[i]=bg_z(k,0,0)[0]
        
    
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


# B=31
# sigma=0
# eta=1
# alphaN=1
# n=0
# u=0/(J2meV*10**(-3))
# l=np.sqrt(hbar/(e*B))
# hwc=3*a0**2*g0**2/(2*l**2*g1)
# D10=hwc*(2*g4/g0+Dp/g1) #+Dp/g1
# print(D10*J2meV)
# H1=(-EZ*sigma+D10*n-eta*u/2*alphaN)*J2meV

# print(H1)

# evals, evecs = bg_h(30,0/J2meV,0)
# print("energy", evals)

#r=3
#for k in range(mid-r,mid+r+1):
#    print("energy", evals[k])
#    print("polarization", sublattice_polarization(evecs[k]))
#    print("--")

###

# npts=100
# u_range=10/J2meV
# u=np.linspace(-u_range,u_range,npts)
# energy=np.zeros((npts,8))
# for i in range(len(u)):
    # energy[i]=np.ndarray.flatten(sym_breaking(u[i]))*J2meV

# plt.plot(u,energy)

###

E = band_structure(1)

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

# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.xlabel('Energy (meV)')
# plt.ylabel('DOS')
# plt.title('Density of states')
# plt.show()