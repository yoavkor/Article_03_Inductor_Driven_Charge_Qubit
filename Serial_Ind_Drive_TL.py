#
#       Transmon - Serial Inductor Drive
#
#   Import Libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from qutip import *
from Graph_Lib import *
#   Constants
hbar = 1.0545718e-34
q = 1.60217663e-19    #  Electron Charge

#   Junction Parameters
fJ = 7e9              #  Qubit Frequency
wJ = 2*np.pi*fJ       #  Qubit Radial Frequency
Lj = 9.340657942e-9   #  LC Inductance
Cj = 1/(wJ**2*Lj)     #  LC Capacitance
Zj = np.sqrt(Lj/Cj)   #  Qubit Impedance
Ns = 2                #  Number of States

#   Inductor Parameters
Cd = 3.7e-15          #  Equivalent Drive Capacitance
Ld = 1/(wJ**2*Cd)     #  Drive Inductance

#   Transmission line (TL, Bath) Parameters
L = 5                 #  TL length
Z0 = 50.0             #  TL characteristic impedance
vp = 210e6            #  0.7 * C
wB = wJ               #  TL frequency ~ qubit frequency
Wmin = 2*np.pi*6.65e9 # minimum of the Hamiltonian integral instead of 0
Wmax = 2*np.pi*7.35e9 # maximum of the Hamiltonian integral instead of infty
Nfm = 2               # Number of discretized frequencies 
Nfs = 2               # Number of fock states

Wedges = np.linspace(Wmin, Wmax, Nfm + 1)       #  (Omega Edges) Array of values from 0 to Wmax. Discretization of the integral 
Wgrid = 0.5 * (Wedges[:-1] + Wedges[1:])        #  Array of mid points between current Wedge, to next Wedge 
dims = [Nfs]*(2*Nfm) + [Ns]  #  Hilbert-space dimensions list. b and c modes so 2 * Nfm, and qubit modes so + 1

#   Drive Signal Parameters
SigmaW = 0.08         #  Control Signal Relative Std. Deviation
Ntm = 1001             #  Number of Time Points
ThetaDdeg = 60.       #  Rotation Drive Angle
PhiDdeg = 0.          #  Rotation Axis Azimuth Angle

ThetaD = math.radians(ThetaDdeg)
PhiD = math.radians(PhiDdeg)

# QuTiP/ZVODE is more stable when the time and Hamiltonian scales are
# numerically comparable.  The model uses seconds and angular frequencies
# (rad/s), so the solver grid is converted to ns below.
TimeScale = 1e-9
Options = {"method": "adams", "atol": 1e-8, "rtol": 1e-6,
           "max_step": 0.01, "nsteps": 100000}

#   Embedding the Operators in Full Hilbert Space
def embed_op(local_op, mode_index, dims):       #  Embed each operator into full Hilbert space
    ops = [qeye(d) for d in dims]
    ops[mode_index] = local_op
    return tensor(ops)

#   Annihilation Operator  
a = embed_op(destroy(Ns), 2*Nfm, dims)     #  Annihilation Operator in full Hilbert space

#   Initialize Bath Modes
b_modes = []
c_modes = []

for k in range(Nfm):
    b_modes.append(embed_op(destroy(Nfs), k, dims))   #  Annihilation operator for forward TL modes
    c_modes.append(embed_op(destroy(Nfs), k + Nfm, dims))   #  Annihilation operator for reflected TL modes


#
#   Qubit Hamiltonian (105)(b)
#
H0 = wJ * (a.dag() * a + 0.5)         #  Qubit Hamiltonian, (3-163)



#
#   Interaction Hamiltonian (105)(d)
#
H_I = 0

for omega_k, b_k, c_k in zip(Wgrid, b_modes, c_modes):
    H_I += (b_k.dag() + b_k + c_k.dag() + c_k) / np.sqrt(omega_k)

H_I = H_I * (-1)/(2*Ld)*np.sqrt(Zj*Z0/2*np.pi) * (a.dag() + a)



#
#   TL (Bath) Hamiltonian (105)(c)
#
H_B = 0

for omega_k, b_k, c_k in zip(Wgrid, b_modes, c_modes):   #  Build Bath Hamiltonian
    H_B += omega_k * (b_k.dag() * b_k + c_k.dag() * c_k)



#
#   Drive Hamiltonian (105)(e)
#
wD = wJ
delW = wJ - wD                  #  Freq. Diffrence (3-267)
SigmaT = 1/(SigmaW*wJ)          #  Gaussian Pulse Standard Deviation
Nts = math.floor(5*SigmaT*fJ)   #  Number of Time Cycles
Time = np.linspace(-Nts/fJ,Nts/fJ,Ntm)         #  Time Vector
SigmaP = np.sqrt(hbar*Zj)       #  Magnetic Flux ZPF, (2-102)
SigmaQ = np.sqrt(hbar/Zj)       #  Charge ZPF, (2-102)

Vs = SigmaQ*wD*Ld/np.sqrt(np.pi)/SigmaT/(1-delW/1.9/wD) * np.exp(0.5*(delW*SigmaT)**2)      #  Source Voltage (3-296)

#   Drive Signal Function
def drive(t) :
    vout = np.exp(-0.5*(t/SigmaT)**2)*np.sin(wD*t-PhiD)
    return vout

#   Drive Shift Freq.
IntD = np.zeros(Ntm)
for iT in range(1,Ntm):
    Tvec = Time[:(iT+1)]
    Vvec = drive(Tvec)
    Ires = integrate.trapezoid(Vvec, Tvec)
    IntD[iT] = Ires

OmegaD = -Vs/(np.sqrt(2)*SigmaQ*Ld)*IntD        #  Driving Signal, (3-297)
OmegaD_ns = OmegaD
H_D = 0

for omega_k, b_k, c_k in zip(Wgrid, b_modes, c_modes):   #  Build Bath Hamiltonian
    H_D += np.sqrt(omega_k) * ((b_k.dag() + b_k) * np.exp(-1j*omega_k*L/vp) - (c_k.dag() + c_k) * np.exp(1j*omega_k*L/vp))

# Quantized transmission-line voltage normalization.  Using 1/sqrt(hbar)
# makes the drive operator numerically enormous and destabilizes ZVODE.
H_D = H_D * np.sqrt(hbar*Z0/(2*np.pi))
H_D = [H_D, -OmegaD_ns]


#
#   Total Hmiltonian (105)(a)
#
#  H = H_B + H_I + H0
H = H0
H = [H, H_D]

#   Master Equation Solver
psi_list = [basis(Nfs,0)]*(2*Nfm)
psi_list.append(basis(Ns,0))
psi0 = tensor(psi_list)         #  Initial State = Ground State
ProbT = []                      #  Define Result Array
AbsT = []                       #  Define Abs. Value Array
AngT = []                       #  Define Phase Array
BlochT = []                     #  Bloch Angles Array
lstL = []                       #  Create Labels List


print("\n\nH[0] shape")
print(H[0].shape)
print("\n\nH[0].norm()")
print(H[0].norm())
print("\n\nH[0].isherm")
print(H[0].isherm)
print("\n\nTime")

print(Time[0], Time[-1], len(Time))

# Shift the grid to start at zero and express it in ns.  The coefficient array
# remains aligned point-for-point with this grid, while the Hamiltonian is
# converted from rad/s to rad/ns.
Time_solver = (Time - Time[0]) / TimeScale
H_solver = [H[0] * TimeScale,
            [H[1][0] * TimeScale, H[1][1]]]

psiT = mesolve(H_solver, psi0, Time_solver, [], options=Options)  # Solve Master Equation
States = psiT.states            #  Extract States Matrix



