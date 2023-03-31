def bg_e(B,u):
    lB=np.sqrt(hbar/(e*B)) # meters
    hwc=3*a**2*g0**2/(2*lB**2*g1)
    
    energy=np.zs((2*dim-2))
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
    

#for i in range(0,npts):
    # energy_e[i] = bg_e(B_[i],0) # meV
    #energy_h[i], eigvec = bg_h(B_[i],0,21) # meV

# eigenenergies at B_[0]
# print(energy_h[0])
#print(energy_h[10,20],energy_h[10,21],energy_h[10,22])

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
LLw=np.zs((len(x),len(y)))
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

#npts = 100
#B_ = np.linspace(0.1,10,npts) # Tesla
# energy_e = np.zs((npts,2*dim-2)) # 2*dim-2 for e, 4*dim for h
#energy_h = np.zs((npts,4*dim))
#eigvec = np.zs(4*dim)

# print(energy_h1[z],energy_h1[z+1],energy_h1[z+2],energy_h1[z+3])
# print(eigvec)