import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib import cm as cmx

import time
import argparse
import warnings
# from multiprocessing import Process, Pool, Manager
import multiprocessing as mp

plt.rcParams.update({'font.size': 15})
plt.rc('axes', labelsize=20) 
fig, ax = plt.subplots()

### paramters (SI units)

dim=500 # dimension of angular momentum space
mid=int((4*dim-1)/2)-1

hbar=1.054571*10**(-34) # planck's constant (J*s)
e=1.60217663*10**(-19) # electron charge (C)
J2meV=10**3/e # joules to meV conversion factor
a0=2.46*10**(-10) # graphene AA distance
# m=9.1093837*10**(-31) # electron mass

i=complex(0,1)

### Paramters

# Hopping terms (J)
g0=2.61*10**3/J2meV
g1=0.361*10**3/J2meV
g3=0#-0.283*10**3/J2meV
g4=-0.138*10**3/J2meV

# symmetry-breaking
# EZ=3.58/J2meV # Zeeman splitting, breaks spin symmetry
Dp = 0.015*10**3/J2meV # on-site energy due to hBN, breaks PH symmetry

# 
v0 = np.sqrt(3)*a0*g0/(2*hbar)
# v4 = np.sqrt(3)*a0*np.abs(g4)/(2*hbar)

# Landau ladder operators
def a_creat():
    # return the matrix form of the creation operator
    X = np.zeros((dim,dim))
    for i in range(dim-1):
        X[i+1,i] = np.sqrt(i+1)
    
    return np.matrix(X)

def a_annih():
    # return the matrix form of the annihilation operator
    return np.transpose(a_creat())

### Initialize operators
id_d=np.identity(dim)
z=np.zeros((dim,dim))
ad=a_creat()
a=a_annih()

###

def hex_grid(pt_den,r,psi):
    R=2/np.sqrt(3)*r
    total_pts=int(3*pt_den**2)
    k=[[],[]]
    step=r/pt_den
    global dk2
    dk2=step**2
    
    # rectangle
    y=step/2
    i=0
    while (y<=R/2):
        n=int(np.ceil(r/step))
        k[1]=np.append(k[1],np.array([y]*n))
        k[0]=np.append(k[0],np.arange(step/2,r,step))
        i=i+pt_den
        y=y+r/pt_den
        
    # triangle
    x=r
    i=0
    while (y<=R):
        n=int(np.ceil((x-step/2)/step))
        k[1]=np.append(k[1],np.array([y]*n))
        k[0]=np.append(k[0],np.arange(step/2,x,step))
        i=i+n
        y=y+r/pt_den
        x=x-r/pt_den*np.sqrt(3)
    
    k[1]=np.append(k[1],-k[1])
    k[0]=np.append(k[0],k[0])
    
    k[1]=np.append(k[1],k[1])
    k[0]=np.append(k[0],-k[0])
    
    sites=np.zeros((len(k[0]),2))

    M_psi=np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]])
    for i in range(len(k[0])):
        sites[i]=M_psi@np.array([k[0][i],k[1][i]])
        
    return sites

def f(k):
    s=a0/(2*np.cos(np.pi/6))
    d1=s*np.array([0,1])
    d2=s*np.array([-np.sqrt(3)/2,-1/2])
    d3=s*np.array([np.sqrt(3)/2,-1/2])
    return np.exp(-i*np.dot(d1,k))+np.exp(-i*np.dot(d2,k))+np.exp(-i*np.dot(d3,k))

### Bilayer Graphene Hamiltonians

def bg_B0(eta,k,u):
    # low energy
    # bilayer graphene in zero magnetic field (single particle)
    # u: electric field between layers
    # return eigenstate of nth level, and all energy levels
    # eta=+-1 for K,K' respectively

    # pi=k[0]-i*k[1]
    pi=f(k)
    pid=np.conjugate(pi)

    h=J2meV*np.array([[eta*u/2,g3*pi,g4*pid,g0*pid],
                      [g3*pid,-eta*u/2,g0*pi,g4*pi],
                      [g4*pi,g0*pid,(-eta*u/2+Dp),g1],
                      [g0*pi,g4*pid,g1,(eta*u/2+Dp)]])

    eigenvalue, eigenvector=np.linalg.eig(h)
    eigenvector=np.transpose(eigenvector)
    eig_vecs_sorted=eigenvector[eigenvalue.argsort(),:]
    eig_vals_sorted=np.sort(eigenvalue)
    
    return eig_vals_sorted, np.array(eig_vecs_sorted)

def bg_BN0(eta,B,u):
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
   
    h=J2meV*np.block([[eta*u/2*id_d,w3*a,w4*ad,w0*ad],
                     [w3*ad,-eta*u/2*id_d,w0*a,w4*a],
                     [w4*a,w0*ad,(-eta*u/2+Dp)*id_d,g1*id_d],
                     [w0*a,w4*ad,g1*id_d,(eta*u/2+Dp)*id_d]])

    eigenvalue, eigenvector=np.linalg.eig(h)
    eigenvector=np.transpose(eigenvector)
    eig_vecs_sorted=eigenvector[eigenvalue.argsort(),:]
    eig_vals_sorted=np.sort(eigenvalue)
    
    # remove high-level states (there is no maximum state)
    
    issue=np.zeros((0),dtype=int)
    k=0
    thsld=10**-5
    for k in range(len(eig_vecs_sorted)):
        for j in range(1,5):
            if (np.abs(eig_vecs_sorted[k,j*dim-1])>thsld):
                issue=np.append(issue,int(k))
    
    if (len(issue)!=0):
        eig_vecs_sorted=np.delete(eig_vecs_sorted,issue,0)
        eig_vals_sorted=np.delete(eig_vals_sorted,issue,0)
    
    return eig_vals_sorted, eig_vecs_sorted

### Auxiliary functions

def layerpolarization(a,eta,Bonoff,opt=0):
    # layer polarization (alpha)
    
    if (Bonoff):
        if (eta==1):
            A=np.max(np.abs(a[0:dim]))
            Bp=np.max(np.abs(a[dim:2*dim]))
            Ap=np.max(np.abs(a[2*dim:3*dim]))
            B=np.max(np.abs(a[3*dim:]))
        if (eta==-1):
            Bp=np.max(np.abs(a[0:dim]))
            A=np.max(np.abs(a[dim:2*dim]))
            B=np.max(np.abs(a[2*dim:3*dim]))
            Ap=np.max(np.abs(a[3*dim:]))
    else:
        if (eta==1):
            A=np.abs(a[0])
            Bp=np.abs(a[1])
            Ap=np.abs(a[2])
            B=np.abs(a[3])
        if (eta==-1):
            Bp=np.abs(a[0])
            A=np.abs(a[1])
            B=np.abs(a[2])
            Ap=np.abs(a[3])
    
    norm=A**2+B**2+Ap**2+Bp**2
    # alpha=(A**2+B**2-Ap**2-Bp**2)/norm
    # beta=(A**2+B**2)/norm
    # beta=B**2/norm
    
    opt=0
    
    #if (opt==0):
    #    beta=A**2/norm
    #else:
    #    beta=B**2/norm
    # beta=1
    return beta

### Density of states

def B0_DOS(pt_den,ue):
    kr=np.around(4*np.pi/(3*a0)*(np.sqrt(3)/2),3) # BG
    sites=hex_grid(pt_den,kr,np.pi/2)
    
    global energy,layerpol,nsites,u,opt
    nsites=len(sites)
    u=ue/(J2meV*10**(-3)) # meV
    # u=0.06/(J2meV*10**(-3)) # meV

    opt=0
    with mp.Pool(processes=16) as pool:
        result=np.array(pool.map(B0_DOS_process,sites))
        energy=np.array(result[:,0],dtype=float)
        layerpol=np.array(result[:,1],dtype=float)
    
    global minE,maxE,r
    r=200
    E=60
    maxE=E # (meV)
    minE=-E

    with mp.Pool(processes=16) as pool:
        DOS=np.array(pool.map(B0_plot_process,range(r)))
    # DOS=DOS*(dk2/(2*np.pi)**2) # confused about this
    
    ### chemical potential as a function of gate voltage
    energy_range=np.linspace(minE,maxE,r)
    plt.plot(energy_range,DOS,color="blue",label="A")
    
    opt=1
    with mp.Pool(processes=16) as pool:
        result=np.array(pool.map(B0_DOS_process,sites))
        energy=np.array(result[:,0],dtype=float)
        layerpol=np.array(result[:,1],dtype=float)
    
    with mp.Pool(processes=16) as pool:
        DOS2=np.array(pool.map(B0_plot_process,range(r)))
    plt.plot(energy_range,DOS2,color="red",label="B")
    plt.legend()
    
    # VG=np.zeros((r))
    # dmu=(maxE-minE)/r
    # VG[0]=DOS[0]*dmu
    # for m in range(1,r):
        # VG[m]=VG[m-1]+DOS[m]
    # VG=dmu*VG
    # VG=VG-VG[99] # VG(mu=0)=0
    # unique=np.unique(VG,return_index=True)
    # VG_unique=unique[0]
    # VG_indices=unique[1]
    # mu=energy_range[VG_indices]
    # plt.plot(VG_unique,mu)
    # plt.ylabel("Chemical potential (meV)")
    # plt.xlabel("Carrier density")
    
    
    plt.xlabel('Energy (meV)')
    plt.ylabel('DOS')
    # plt.title('Density of states')

    #name=str(pt_den)
    name="A"
    plt.savefig("./plots/"+name+".png",bbox_inches="tight",dpi=300)

def B0_DOS_process(k):
    evalsP,evecsP=bg_B0(1,k,u)
    evalsM,evecsM=bg_B0(-1,k,u)
    energy=np.append(evalsP,evalsM)
    layerpol=np.zeros((8))
    for e in range(4):
        layerpol[e]=layerpolarization(evecsP[e],1,0,opt) # valley +1
        layerpol[e+4]=layerpolarization(evecsM[e],-1,0,opt) # valley -1
    return np.array(energy), np.array(layerpol)

def B0_plot_process(d):
    global minE,maxE,r,nsites
    emin=minE+d/r*(maxE-minE)
    emax=minE+(d+1)/r*(maxE-minE)
    DOS=0
    for e in range(8):
        for m in range(nsites):
            if ((energy[m,e]<emax) & (energy[m,e]>=emin)):
                DOS+=layerpol[m,e]
    return DOS
    
def BN0_DOS(B,u):
    global energy,layerpol,nevalsM,nevalsP
    u=u/(J2meV*10**(-3))
    # u=0.06/(J2meV*10**(-3)) # meV
    evalsP,evecsP=bg_BN0(1,B,u)
    evalsM,evecsM=bg_BN0(-1,B,u)
    energy=np.append(evalsP,evalsM) # np.sort messes DOS up!
    nevalsP=len(evalsP)
    nevalsM=len(evalsM)
    layerpol=np.zeros((nevalsP+nevalsM))

    for e in range(nevalsP):
       layerpol[e]=layerpolarization(evecsP[e],1,1) # valley +1
    for e in range(nevalsM):
        layerpol[nevalsP+e]=layerpolarization(evecsM[e],-1,1) # valley -1

    # energy=np.sort(energy) # REMOVE
    # for m in range(len(energy)):
        # e=energy[m]
        # # lp=layerpol[m]
        # if (np.abs(e)<50):
            # print(e)

    global minE,maxE,r
    r=1000 # 200
    E=150 # 80 # (meV)
    maxE=E
    minE=-E

    with mp.Pool(processes=16) as pool:
        DOS=np.array(pool.map(BN0_plot_process,range(r)))
    
    # return DOS,energy
    
    # energy_range=np.linspace(minE,maxE,r)
    # plt.plot(energy_range,DOS,color="blue",label="A")
    
    # for e in range(nevalsP):
       # layerpol[e]=layerpolarization(evecsP[e],1,1,opt=1) # valley +1
    # for e in range(nevalsM):
        # layerpol[nevalsP+e]=layerpolarization(evecsM[e],-1,1,opt=1) # valley -1
        
    # with mp.Pool(processes=16) as pool:
        # DOS2=np.array(pool.map(BN0_plot_process,range(r)))
    
    # plt.plot(energy_range,DOS2,color="red",label="B")
    # plt.legend()
    
    name="A"
    plt.xlabel('Energy (meV)')
    plt.ylabel('DOS')
    # plt.title('Density of states')
    plt.savefig("./plots/"+name+".png",bbox_inches="tight",dpi=300)
    
def BN0_plot_process(d):
    global minE,maxE,r
    emin=minE+d/r*(maxE-minE)
    emax=minE+(d+1)/r*(maxE-minE)
    DOS=0
    for e in range(nevalsP+nevalsM):
        if ((energy[e]<emax) and (energy[e]>=emin)):
            DOS+=layerpol[e]
    return DOS

def B0_bandline(pt_den):

    u=0/(J2meV*10**(-3))
    kr=0.5
    energy=np.zeros((pt_den,4))
    
    k=np.linspace(-kr,kr,pt_den)
    for m in range(pt_den):
        BG=bg_B0(1,[k[m],0],u) # valley=+1
        energy[m]=BG[0]
    
    plt.plot(k,energy)
    plt.xlabel(r"$k_x$ (1/m)")
    plt.ylabel(r"Energy (meV)")
    
    name="A"
    plt.savefig("./plots/"+name+".png",bbox_inches="tight",dpi=300)

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-pt_den", type=int, default=500)
    parser.add_argument("-B", type=float, default=6)
    parser.add_argument("-u", type=float, default=0)

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    
    t = time.time()
    B0_DOS(args.pt_den,ue=args.u)
    # BN0_DOS(B=args.B,u=args.u)
    # B0_bandBZ(0,args.pt_den)
    # B0_bandline(args.pt_den)
    print(time.time()-t)