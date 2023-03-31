import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams.update({'font.size': 15})
plt.rc('axes', labelsize=20) 
fig, ax = plt.subplots()

###

dim = 10
hbar = 1.05*10**(-34) # J*s
e = 1.602*10**(-19) # Coulomb
J2meV = 10**3/e

# parameters (SI units)
g0 = 2.61*e
g1 = 0.361*e
g3 = 0#-0.283*e
g4 = 0#-0.138*e
Dp = 0.015*e

a = 1.42*10**(-10) # m
vF = np.sqrt(3)/2*g0*a/hbar # J*m
vF = 2*vF

def a_creat():
    # return the matrix form of the annihilation operator
    # the creation operator is the hermitian-conjugate
    a_creation = np.zeros((dim,dim))
    for i in range(dim-1):
        a_creation[i+1,i] = np.sqrt(i+1)
    
    return np.matrix(a_creation)

def a_annih():
    return np.transpose(a_creat())
    
def graphene_h(B):
    
    lB = np.sqrt(hbar/(e*B))
    
    zero = np.zeros((dim,dim))
    a_annih_op = a_annih()
    a_creat_op = a_creat()
    
    h = hbar*vF*np.sqrt(2)/lB*np.block([[zero,a_creat_op],[a_annih_op,zero]])
    eigenvalue, eigenvector = np.linalg.eig(h)
    return np.sort(eigenvalue)

def bgraphene_h(B,u):

    lB = np.sqrt(hbar/(e*B)) # meters
    w0 = hbar*vF*np.sqrt(2)/lB
    #w4 = g4/g0*w0
    #w3 = g3/g0*w0
    w4=0
    w3=0
    eta = 1 # +-1 for K,K'
    
    id = np.identity(dim)
    a_annih_op = a_annih()
    a_creat_op = a_creat()
    
    h = np.block([[eta*u/2*id,w3*a_annih_op,w4*a_creat_op,w0*a_creat_op],
                  [w3*a_creat_op,-eta*u/2*id,w0*a_annih_op,w4*a_annih_op],
                  [w4*a_annih_op,w0*a_creat_op,(-eta*u/2+Dp)*id,g1*id],
                  [w0*a_annih_op,w4*a_creat_op,g1*id,(eta*u/2+Dp)*id]])

    eigenvalue, eigenvector = np.linalg.eig(h)
    return np.sort(eigenvalue)

def bgraphene_energy(B,u):
    lB = np.sqrt(hbar/(e*B)) # meters
    w0 = hbar*vF*np.sqrt(2)/lB
    
    energy = np.zeros((4*dim))
    for N in range(-2*dim,2*dim):
        energy[N] = (N/np.abs(N))*w0**2/g1*np.sqrt(np.abs(N)*(np.abs(N)-1)+(u*g1/(2*w0**2))**2)
    return energy
    
def band2(B,u):
    lB = np.sqrt(hbar/(e*B)) # meters
    w0 = hbar*vF*np.sqrt(2)/lB
    zero = np.zeros((dim,dim))
    id = np.identity(dim)
    a_annih_op = a_annih()
    a_creat_op = a_creat()
    h0 = -w0**2/g1*np.block([[u/2*id,a_creat_op@a_creat_op],[a_annih_op@a_annih_op,-u/2*id]]) + np.block([[u/2*id,zero],[zero,-u/2*id]])
    h1 = w0**2/g1*(np.abs(g4)/g0+Dp/g1)*np.block([[a_creat_op@a_annih_op,zero],[zero,a_annih_op@a_creat_op]])
    h = h0 + h1
    eigenvalue, eigenvector = np.linalg.eig(h)
    return np.sort(eigenvalue)
    
npts = 100
B_ = np.linspace(0,20,npts) # Tesla
energy = np.zeros((npts,2*dim)) # 2 for graphene, 4 for bg

for i in range(0,npts):
    energy[i] = graphene_h(B_[i])*J2meV # meV
    # energy[i] = bgraphene_h(B_[i],0)*J2meV # meV
    # energy[i] = bgraphene_energy(B_[i],0)*J2meV
    # energy[i] = band2(B_[i],0)*J2meV

plt.plot(B_,energy)
#plt.ylim([-70,70])
plt.ylim([-200,200])

#u_ = np.linspace(-20,20,npts)/J2meV
#for i in range(0,npts):
#    energy[i] = bgraphene_h(15,u_[i])*J2meV # meV
    
#plt.plot(u_*J2meV,energy)
#plt.ylim([-40,40])
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.show()