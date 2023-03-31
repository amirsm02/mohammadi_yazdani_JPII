import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy.special import factorial as fact

plt.rcParams.update({'font.size': 15})
plt.rc('axes', labelsize=20) 
fig, ax = plt.subplots()

### paramters (SI units)

dim = 20
hbar = 1.05*10**(-34) # J*s
e = 1.602*10**(-19) # Coulomb
J2meV = 10**3/e
a = 2.46*10**(-10) # m

# J
g0 = -2.61*10**3/J2meV
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
    # annihilation operator
    return np.transpose(a_creat())

def bg_h(B,u,N):
    # bilayer graphene in magnetic field (single particle)
    # u: electric field between layers
    # return eigenstate of Nth level and all energy levels

    hw0=30.6*np.sqrt(B)/J2meV # J
    # Dp=0
    
    id_d=np.identity(dim)
    zero=np.zeros((dim,dim))
    a=a_annih()
    ad=a_creat()
    
    h=J2meV*hw0*np.block([[u/(2*hw0)*id_d,zero,ad,-g4/g0*ad],
                            [zero,-u/(2*hw0)*id_d,-g4/g0*a,a],
                            [a,-g4/g0*ad,(u/2+Dp)/hw0*id_d,g1/hw0*id_d],
                            [-g4/g0*a,ad,g1/hw0*id_d,(-u/2+Dp)/hw0*id_d]])

    eigenvalue, eigenvector=np.linalg.eig(h)
    eigenvector=np.transpose(eigenvector)
    eig_vecs_sorted=eigenvector[eigenvalue.argsort(),:]
    
    return np.sort(eigenvalue), eig_vecs_sorted[N]

def bg_e(B,u):
    lB=np.sqrt(hbar/(e*B)) # meters
    hwc=3*a**2*g0**2/(2*lB**2*g1)
    
    energy=np.zeros((2*dim-2))
    for N in range(0,2*dim-2):
        M=N+2
        energy[N]=hwc*np.sqrt(M*(M-1))
    return energy*J2meV

def LL(B,n,m,lB,r,phi):
    # m any integer
    # n non-negative integer
    # meters
    N_nm=np.sqrt(fact(n)/(lB**2*2**(np.abs(m))*fact(n+np.abs(m))))
    R_nm=N_nm*(r/lB)**(np.abs(m))*np.exp(-r**2/(4*lB**2))*scipy.special.assoc_laguerre(r**2/(2*lB**2), n, k=np.abs(m))
    Phi=np.exp(i*m*phi)/np.sqrt(2*np.pi)
    return R_nm*Phi
    
npts = 100
B_ = np.linspace(0.1,10,npts) # Tesla
# energy_e = np.zeros((npts,2*dim-2)) # 2*dim-2 for e, 4*dim for h
energy_h = np.zeros((npts,4*dim))
eigvec = np.zeros(4*dim)

#for i in range(0,npts):
    # energy_e[i] = bg_e(B_[i],0) # meV
    #energy_h[i], eigvec = bg_h(B_[i],0,21) # meV

# eigenenergies at B_[0]
# print(energy_h[0])
#print(energy_h[10,20],energy_h[10,21],energy_h[10,22])
zero=int(4*dim/2)
energy_h1, eigvec = bg_h(31,0,zero+2) # meV

print(energy_h1[zero],energy_h1[zero+1],energy_h1[zero+2],energy_h1[zero+3])
print(eigvec)

step=int(dim/4)
A=np.max(np.abs(eigvec[0,0:step]))
Bp=np.max(np.abs(eigvec[0,step:2*step]))
B=np.max(np.abs(eigvec[0,2*step:3*step]))
Ap=np.max(np.abs(eigvec[0,3*step:]))
print("A:",A,"Bp",Bp,"B",B,"Ap",Ap)
print("alpha",A**2-Bp**2+B**2-Ap**2)

norm=0
for i in range(4*dim):
    norm+=eigvec[0,i]**2
print("norm:", norm)

def phi_(x,y):
    if (x>0):
        return np.arctan(y/x)
    else:
        return -np.arctan(y/x)

k=3
B=1 #T
lB = np.sqrt(hbar/(e*B))
x=np.linspace(-k*lB,k*lB,50) #m
y=np.linspace(-k*lB,k*lB,50) #m
LLw=np.zeros((len(x),len(y)))
n=3
m=0
for i in range(len(x)):
    for j in range(len(y)):
        r=np.sqrt(x[i]**2+y[j]**2)
        phi=phi_(x[i],y[j])
        LLw[i,j]=LL(B,n,m,lB,r,phi)
        
#plt.imshow(LLw)
# add colorbar

# plt.plot(B_,energy_e,color="black")
#plt.plot(B_,energy_h,color="red")
#plt.ylim([-200,200])

#u_ = np.linspace(-20,20,npts)/J2meV
#for i in range(0,npts):
#    energy[i] = bgraphene_h(15,u_[i])*J2meV # meV

plt.title("Bilayer graphene energy levels")
plt.xlabel("Magnetic field (T)")
plt.ylabel("Energy (meV)")

ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
#plt.show()