#
#       Transmon - Serial Capacitor Drive
#
#   Import Libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from fontTools.tfmLib import MATHSY
# from numpy.conftest import dtype
from numpy.conftest import hypothesis
import scipy.integrate as integrate
import scipy.special as special
from qutip import *
from Graph_Lib import *
from Qubit_Lib import *
#   Constants
hbar = 1.0545718e-34
#   Circuit Parameters
fJ = 13.5e9         #  Qubit Frequency
wJ = 2*np.pi*fJ     #  Qubit Radial Frequency
Lj = 134e-12        #  LC Inductance
Cj = 1/(wJ**2*Lj)   #  LC Capacitance
Cd = 3.7e-15        #  Equivalent Drive Capacitance
Ld = 1/(wJ**2*Cd)   #  Drive Inductance
Zj = np.sqrt(Lj/Cj) #  Qubit Impedance
#   Drive Signal Parameters
Nphi = 100          #  No. of Points in Flux Range
Pcyc = 200          #  No of Points in fJ Cycle
Phi_max = 6.        #  Normalized Flux Range
Ns = 2              #  Number of States
SigmaW = 0.08       #  Control Signal Relative Std. Deviation
Ntm = 501           #  Number of Time Points
#
ThetaDdeg = 60.     #  Rotation Drive Angle
PhiDdeg = 0.        #  Rotation Axis Azimuth Angle
#
#   Calculate
#
ThetaD = math.radians(ThetaDdeg)
PhiD = math.radians(PhiDdeg)
#   Driving Signal
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
OmegaD = -Vs/(np.sqrt(2)*SigmaQ*Ld)*IntD     #  Driving Signal, (3-297)
#   Hamiltonian Operators
a = destroy(Ns)                 #  Annihilation Operator
H0 = wJ*(a.dag() * a + 0.5)     #  Qubit Hamiltonian, (3-163)
Qd = (a.dag() + a)             #  Phase Operator, (3-163)
Hd = [Qd, -OmegaD]               #  Drive Operator
H = [H0, Hd]                    #  Total Hamiltonian Operator, (3-163)
#   Master Equation Solver
psi0 = basis(Ns,0)           #  Initial State = Ground State
ProbT = []      #  Define Result Array
AbsT = []       #  Define Abs. Value Array
AngT = []       #  Define Phase Array
BlochT = []     #  Bloch Angles Array
lstL = []       #  Create Labels List
psiT = mesolve(H, psi0, Time, [])  # Solve Master Equation
States = psiT.states            #  Extract States Matrix
#
#   Post Processing
#
for jS in range(Ns):
    lstL.append('State ' + str(jS))  # Append Label
    Amps = np.array([state.full()[jS][0] for state in States])
    Aabs = np.abs(Amps)         #  Absolute Value of State
    Prob = np.abs(Amps)**2      #  Extract Probability
    Ang = np.angle(Amps)        #  Extract Phase
    AbsT.append(Aabs)           #  Add Abs. Value to Matrix
    ProbT.append(Prob)          #  Add Probability to Matrix
    AngT.append(Ang)            #  Add Phase to Matrix
Vd = Vs * np.exp(-0.5*(Time/SigmaT)**2) * np.sin(wD*Time-PhiD)          #  Driving Signal
#
AbsT = np.transpose(AbsT)
AngT = np.transpose(AngT)
ProbT = np.transpose(ProbT)
ProbT0 = ProbT[0][0] + ProbT[0][1]          #  Start Probability
ProbTE = ProbT[-1][0] + ProbT[-1][1]        #  End Probability
ThetaE = 2 * np.acos(np.sqrt(ProbT[-1][0]))     #  Resulted Mix Angle (rad)
ThetaEdeg = math.degrees(ThetaE)            #  resulted Mix Angle (deg)
AngT = np.unwrap(AngT,axis=0)               #  Unwrap State Phase
(Freq_Rel, Phase_Rel) = States_Freq_Phase (wJ,Time,AngT)
(Aelev, Aazim) = Bloch_Sphere_Angles (AbsT,Phase_Rel)      #  Bloch Sphere Angles
#
#   Approximated Two States Solution
Ud = np.array([[np.cos(ThetaD/2), 1j*np.exp(-1j*PhiD)*np.sin(ThetaD/2)],       #  Evolution Matrix
               [1j*np.exp(1j*PhiD)*np.sin(ThetaD/2), np.cos(ThetaD/2)]])       #  eq. (3-183)
if Ns==2 :
    psiI = psi0.full()
else :
    psiC = psi0.full()
    psiI = psiC[:2]
psiE = np.matmul(Ud, psiI)
Ang0 = np.angle(psiE[0])
psiE = psiE * np.exp(-1j*Ang0)
apxAzim = np.angle(psiE[1],deg=True)
apxElev = 2*np.rad2deg(np.atan2(abs(psiE[1]), abs(psiE[0])))
#
#   Plot
#
#   Drive Signal
plt.figure(num=1, figsize=(8, 5.6))
(rVmax,sVmax,kVmax) = Num_Disp(max(Vd),1)
plt.plot(Time*1e9, Vd*kVmax)
Glist = ['Drive Signal','t (nsec)','$V_d$  ('+ sVmax + 'V)']
Graph_Title(Glist)
Plist = [('Data',0.0,'',''), ('$F_J$',fJ,'%5.1F','Hz'), ('$V_{pk}$',Vs,'%4.1F','V'), ('$\\sigma_t$',SigmaT,'%5.0F','sec')]
Titles (Plist, 1, 1)
# plt.show()
#
#   Drive Frequency Shift
colors = mcolors.TABLEAU_COLORS
names = list(colors)
cColor = colors[names[0]]
fig, ax1 = plt.subplots(figsize=(8, 5.6))
(rTime,sTime,kTime) = Num_Disp(max(Time),1)
(rVmax,sVmax,kVmax) = Num_Disp(max(Vd),1)
FreqD = OmegaD/(2*np.pi)
(rFrq,sFrq,kFrq) = Num_Disp(max(FreqD),1)
ax1.plot(Time*kTime, Vd*kVmax)
Glist = ['Drive Frequency Shift',('t ('+ sTime +'sec)'),'$V_d$  ('+ sVmax + 'V)']
Graph_Title(Glist)
ax1.tick_params(axis='y', labelcolor=cColor)
ax1.set_ylabel('$V_d$  ('+ sVmax + 'V)',color=cColor)
#
cColor = colors[names[1]]
ax2 = ax1.twinx()
ax2.set_ylabel('$F_d$  ('+ sFrq + 'Hz)', color=cColor, fontsize=14)
ax2.plot(Time*kTime, FreqD*kFrq, color=cColor)
ax2.tick_params(axis='y', labelcolor=cColor)
#
Ticks_Equate(ax1,ax2,0)
Plist = [('Data',0.0,'',''), ('$F_J$',fJ,'%5.1F','Hz'), ('$V_{pk}$',Vs,'%4.1F','V'), ('$\\sigma_t$',SigmaT,'%5.0F','sec')]
Titles (Plist, 1, 1, Ndiv=20)
plt.legend(loc='best')
# plt.show()
#
#   State Probability vs. Time
plt.figure(num=3, figsize=(8, 5.6))
plt.plot(Time*1e9, ProbT, label=lstL)
Glist = ['State Probability','t (nsec)','Probability']
Graph_Title(Glist)
Plist = [('Data',0.0,'',''), ('$F_J$',fJ,'%5.1F','Hz'), ('$V_{pk}$',Vs,'%4.1F','V'), ('$\\sigma_t$',SigmaT,'%5.0F','sec')]
Titles (Plist, 1, 1)
Rlist = [('Mix Angle',0.0,'',''), ('Target',ThetaDdeg,'%4.1F','$\\circ$'), ('Result',ThetaEdeg,'%4.1F','$\\circ$')]
Titles (Rlist, 1, 21)
Clist = [('Total Probability',0.0,'',''), ('Start',ProbT0,'%5.3f',''), ('End',ProbTE,'%5.3f','')]
Titles (Clist, 18, 21)
Blist = [('Init Probability',0.0,'','')]
for iS in range(Ns) :
    Blist.append(('$p'+str(iS)+'_0$',float(ProbT[0,iS]),'%5.3f',''))
Titles (Blist, 1, 12, cColor=[-1,0,0])
Clist = [('End Probability',0.0,'','')]
for iS in range(Ns) :
    Clist.append(('$p'+str(iS)+'_e$',float(ProbT[-1,iS]),'%5.3f',''))
Titles (Clist, 18, 12, cColor=[-1,0,0])
plt.legend(loc='best')
# plt.show()
#
#   State Phase vs. Time
plt.figure(num=4, figsize=(8, 5.6))
plt.plot(Time*1e9, AngT, label=lstL)
Glist = ['State Phase','t (nsec)','Phase (rad)']
Graph_Title(Glist)
Plist = [('Data',0.0,'',''), ('$F_J$',fJ,'%5.1F','Hz'), ('$V_{pk}$',Vs,'%4.1F','V'), ('$\\sigma_t$',SigmaT,'%5.0F','sec')]
Titles (Plist, 1, 1)
Alist = [('Rel. Freq.',0.0,'','')]
for iS in range(Ns) :
    Alist.append(('F'+str(iS),Freq_Rel[iS],'%5.4f',''))
Titles (Alist, 1, 11, cColor=[-1,0,0])
plt.legend(loc='best')
# plt.show()
#
#   Relative State Phase vs. Time
plt.figure(num=5, figsize=(8, 5.6))
Phase_RelD = np.rad2deg(Phase_Rel)
plt.plot(Time*kTime, Phase_RelD, label=lstL)
Glist = ['Relative Eigen State Phase',('t ('+ sTime +'sec)'),'Phase (deg)']
Graph_Title(Glist)
Plist = [('Data',0.0,'',''), ('$F_J$',fJ,'%5.1F','Hz'), ('$V_{pk}$',Vs,'%4.1F','V'), ('$\\sigma_t$',SigmaT,'%5.0F','sec')]
Titles (Plist, 1, 1)
Alist = [('Rel. Freq.',0.0,'','')]
for iS in range(Ns) :
    Alist.append(('F'+str(iS),Freq_Rel[iS],'%5.4f',''))
Titles (Alist, 10, 1, cColor=[-1,0,0])
Blist = [('Init Phase',0.0,'','')]
for iS in range(Ns) :
    Blist.append(('$\\phi$'+str(iS),float(Phase_RelD[0,iS]),'%5.1f','$\\degree$'))
Titles (Blist, 1, 12, cColor=[-1,0,0])
Clist = [('End Phase',0.0,'','')]
for iS in range(Ns) :
    Clist.append(('$\\phi$'+str(iS),float(Phase_RelD[-1,iS]),'%5.1f','$\\degree$'))
Titles (Clist, 18, 12, cColor=[-1,0,0])
AazimD = np.rad2deg(Aazim)
AazimDE = AazimD[-1] % 360
AazimDE = AazimDE - 360*np.heaviside(AazimDE-180,0.0)
Dlist = [('End Azimuth Angle',0.0,'',''), ('$\\phi_e$', AazimDE,'%5.1f','$\\degree$')]
Titles (Dlist, 10, 5)
plt.legend(loc='best')
# plt.show()
#
#   Bloch Sphere Angles
colors = mcolors.TABLEAU_COLORS
names = list(colors)
cColor = colors[names[0]]
fig, ax1 = plt.subplots(figsize=(8, 5.6))
AelevD = np.rad2deg(Aelev)
AazimD = np.rad2deg(Aazim)
FreqD = OmegaD/(2*np.pi)
(rTime,sTime,kTime) = Num_Disp(max(Time),1)
ax1.plot(Time*kTime, AelevD)
Glist = ['Bloch Sphere Angles',('t ('+ sTime +'sec)'),'Elevation Angle (deg)']
Graph_Title(Glist)
ax1.tick_params(axis='y', labelcolor=cColor)
ax1.set_ylabel('Elevation Angle (deg)',color=cColor)
#
cColor = colors[names[1]]
ax2 = ax1.twinx()
ax2.set_ylabel('Azimuth Angle (deg)', color=cColor, fontsize=14)
ax2.plot(Time*kTime, AazimD, color=cColor)
ax2.tick_params(axis='y', labelcolor=cColor)
#
Ticks_Equate(ax1,ax2,1)
Plist = [('Data',0.0,'',''), ('$F_J$',fJ,'%5.1F','Hz'), ('$V_{pk}$',Vs,'%4.1F','V'), ('$\\sigma_t$',SigmaT,'%5.0F','sec')]
Titles (Plist, 1.5, 1, Ndiv=20)
#
Blist = [('Init Angles',0.0,'',''), ('$\\theta_0$',AelevD[0],'%5.1f','$\\degree$'), ('$\\varphi_0$',AazimD[0],'%5.1f','$\\degree$')]
Titles (Blist, 1, 12, cColor=[-1,0,0])
Clist = [('End Angles',0.0,'',''), ('$\\theta_e$',AelevD[-1],'%5.1f','$\\degree$'), ('$\\varphi_e$',AazimDE,'%5.1f','$\\degree$')]
Titles (Clist, 18, 12, cColor=[-1,0,0])
Clist = [('Target Angles',0.0,'',''), ('$\\theta_e$',apxElev[0],'%5.1f','$\\degree$'), ('$\\varphi_e$',apxAzim[0],'%5.1f','$\\degree$')]
Titles (Clist, 18, 17, cColor=[-1,0,0])
#
plt.show()
