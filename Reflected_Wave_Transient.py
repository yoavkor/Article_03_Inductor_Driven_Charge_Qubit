#
#       Reflected Wave Transient Response
#       Square Wave Envelope Case
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
from scipy.optimize import curve_fit
# from qutip import *
#
from Graph_Lib import *
from Qubit_Lib import *
from Coefficients_Solver import *
#   Constants
hbar = 1.0545718e-34
qe = 1.602e-19
c = 2.99792e8
#   Circuit Parameters
fJ = 7.0e9         #  Qubit Frequency
EjEc = 50           #  Qubit's Energies Ratio
Ld = 25e-9          #  Serial Inductor
Z0 = 50             #  Cable Impedance
vp = 0.7*c          #  Cable Phase Velocity
#   Drive Signal Parameters
sigT = 2.0e-9       #  Gaussian Input Signal Time Deviation
mSig = 5            #  Gaussian Width to Deviation Ratio
fD = 6.95e9         #  Drive Frequency
Dw = 1              #  Discretization Number
ThetaD = 60         #  Required Shift Angle (deg)
Te = 2*mSig*sigT    #  Time Vector Length
#Te = 2*sigT    #  Time Vector Length
C0 = 1/(Z0*vp)      #  Cable Capacitance per Length
Lcoax = Te*vp       #  Cable Length
t0 = 0              #  Pulse Initial Time Point
#   Qubit Initial State
P0 = 0.64           #  Ground State Probability
phi1D = 0           #  Excited State Phase
c0 = np.sqrt(P0)
phi1 = np.radians(phi1D)
c1 = np.sqrt(1-P0) * np.exp(1j*phi1)
#   Hamiltonian Parameters
wJ = 2*np.pi*fJ                                 #  Qubit Freq.
Zj = hbar/(np.sqrt(2)*qe**2)/np.sqrt(EjEc)      #  Qubit Impedance
sigQ = np.sqrt(hbar/Zj)                         #  Charge STD
wg = 1/(2*Ld)*np.sqrt(Zj*Z0/(2*np.pi))          #  Interaction Freq.
wD = 2*np.pi*fD                                 #  Drive Frequency
kJ = 2*np.pi*wg**2/wJ                           #  Dumping Constant for Qubit Freq.
kD = 2*np.pi*wg**2/wD                           #  Dumping Constant for Drive Freq.
sigVD = np.sqrt(hbar*wD/(Lcoax*C0))             #  Mantic Flux ZPF for Cable
Theta = ThetaD*np.pi/180                        #  Required Shift Angle
delW = wJ - wD                                  #  Gaussian Shape Pulse Peak Voltage
OmegaD = kJ/2 + 1j*delW                         #  Complex Dumping Factor
gammaJ = kJ/2 + 1j*wJ
delWB = mSig/sigT
Vs = Theta * sigQ*wJ*Ld/(np.sqrt(np.pi)*sigT) * np.exp(-0.5*(delW*sigT)**2)
alphaN = Vs/sigVD                               #  Coherent Wave Amplitude
nd = alphaN**2                                  #  Np. of Photons
fLog = False
#
#   Time Response Functions
#
#   Reflection Signal Time Function
def VdriveT (Time) :
    Vda = -np.sqrt(2)*sigVD
    A1 = alphaN/np.sqrt(2*np.pi)
    A2 = 1-kJ/OmegaD*(1-np.exp(-OmegaD*Time))
    A3 = np.exp(-1j*wD*(Time-t0))
    VoutD = Vda*np.real(A1*A2*A3)
    return VoutD
#
def VtranT (Time) :
    Vda = -np.sqrt(2)*sigVD
    B1 = 1j*np.conj(c0)*c1*kJ/np.sqrt(delWB*kD)
    B2 = np.exp(-gammaJ*Time)
    VoutJ = Vda * np.real(B1*B2)
    return VoutJ
#
def VoutT (Time) :
    Vout = VdriveT(Time) + VtranT(Time)
    return Vout
#
#   Calculate Result
#
#   Reflection Signal - Time Response
Ntm = 1000
Time = np.linspace(0, Te, Ntm+1)
VdT = VdriveT(Time)
VjT = VtranT(Time)
VoT = VoutT(Time)
#
#   Remove Inf and Nan Numbers from Vector
def Remove_InfNan (Y) :
    IndxF = np.where(np.isinf(Y))[0]
    if len(IndxF) > 0 :
        Y[IndxF] = 0.5 * (Y[IndxF - 1] + Y[IndxF + 1])
    IndxA = np.where(np.isnan(Y))[0]
    if len(IndxA) > 0:
        Y[IndxA] = 0.5 * (Y[IndxA - 1] + Y[IndxA + 1])
    return Y
#
#   Frequency Response
def Vbar (w, c0, c1, mD, mJ, Ad=0, AJ=0, fCoeff=False) :
    if Ad==0 :
        Ad = alphaN/np.sqrt(2*np.pi)*(1-kJ/OmegaD)*np.exp(1j*wD*t0)
    if AJ==0 :
        AJ1 = alphaN/np.sqrt(2*np.pi)*kJ/OmegaD*np.exp(1j*wD*t0)
        AJ2 = 1j*np.conj(c0)*c1*kJ/np.sqrt(delWB*kD)
        AJ = AJ1 + AJ2
    F1 = np.conj(Ad)/(w-wD) - Ad/(w+wD)
    F1 = Remove_InfNan(F1)
    F2 = np.conj(AJ)/(kJ/2+1j*(w-wJ)) - AJ/(kJ/2+1j*(w+wJ))
    Fresult = sigVD/np.sqrt(2) * (F1*mD + F2*mJ)
    if not fCoeff:
        return Fresult
    else :
        return Fresult, Ad, AJ
#
#   Spectrum
Nf = 1000
fL = 6e9
fH = 8e9
if fLog :
    f = np.logspace(np.log10(fL), np.log10(fH), Nf+1)
else :
    f = np.linspace(fL, fH, Nf+1)
w = 2*np.pi*f
VfrqD = Vbar(w, c0, c1, 1, 0)
VfrqJ = Vbar(w, c0, c1, 0, 1)
VfrqT = Vbar(w, c0, c1, 1, 1)
#
#   Solve Qubit State coefficients
#
#   Define Freq. Range
Nf = 10000
fnL = 6.85e9
fnH = 7.05e9
Rd_dB = -295
fn = np.linspace(fnL, fnH, Nf+1)
wn = 2*np.pi*fn
#   Find the Equations Constants
(VnT, Ad_exct, AJ_exct) = Vbar(wn, c0, c1, 1, 1, fCoeff=True)
VnTdB = 20*np.log10(np.abs(VnT))        #  Convert Curve to dBV units
wSol = PolySol(wn,VnTdB,Rd_dB)          #  Find the Solution for the f- and f+ Freq.
if len(wSol)==4 :
    wSol = [wSol[0], wSol[3]]           #  Extract the Relevant Solution from the Solutions Vector
fSol = np.asarray(wSol, dtype=float)/(2e9*np.pi)
wN = fSol[0]*2e9*np.pi                  #  Convert f- units to rad/sec
wP = fSol[1]*2e9*np.pi                  #  Convert f+ units to rad/sec
Rd = 10**(Rd_dB/20)                     #  Convert Rd units to linear
fRng = 12e6                             #  Define the Search Range to Locate fJp peak Freq.
wRng = 2*np.pi*fRng                     #  Convert the Search Range to rad/sec units
Indx1 = (np.abs(wn - (wJ-wRng))).argmin()   #  Locate the index of the Range Start
Indx2 = (np.abs(wn - (wJ+wRng))).argmin()   #  Locate the Index of the Range End
IndxJ = Indx1 + np.argmax(np.abs(VnT[Indx1:Indx2]))     #  Locate the fJp Peak Index
RJ = np.abs(VnT[IndxJ])                 #  Extract the RJ Peak Value
RJ_dB = 20*np.log10(np.abs(RJ))         #  Convert the Peak Value to dBV units
wJp = wn[IndxJ]                         #  Extract the Peak Freq.
#   Solve the Approximated Equation
ExtractionResult = extract_qubit_coefficients(omega_d=wD,omega_J=wJ,kappa_J=kJ, omega_plus=wP,omega_minus=wN,
                                              omega_Jp=wJp,R_d=Rd,R_J=RJ,sigma_vd=sigVD)
Ad_apx = ExtractionResult.A_d
AJ_apx = ExtractionResult.A_J
VapxT = Vbar(wn, c0, c1, 1, 1, Ad_apx, AJ_apx)
#
#   Define Model Search Function
def Vbar_Model(wm,ADapxR,ADapxI,AJapxR,AJapxI) :
    ADapx = ADapxR+1j*ADapxI
    AJapx = AJapxR+1j*AJapxI
    complex_output = Vbar(wm, c0, c1, 1, 1, Ad=ADapx, AJ=AJapx)
    return np.hstack([np.real(complex_output), np.imag(complex_output)])
#
#   Curve Fit to Find the Exact Ad and AJ
VnT_real_imag = np.hstack([np.real(VnT), np.imag(VnT)])
p0_guess = [np.real(Ad_apx), np.imag(Ad_apx), np.real(AJ_apx), np.imag(AJ_apx)]
popt, pcov = curve_fit(f=Vbar_Model, xdata=wn, ydata=VnT_real_imag, p0=p0_guess)
Ad_fit = popt[0] + 1j * popt[1]
AJ_fit = popt[2] + 1j * popt[3]
VfitT = Vbar(wn, c0, c1, 1, 1, Ad_fit, AJ_fit)
#
#   State Coefficients
M01 = -1j * (AJ_fit/kJ-Ad_fit/(OmegaD-kJ)) * np.sqrt(delWB*kD)
phi0 = np.angle(M01)
A01 = np.abs(M01)
D01 = np.sqrt(1-4*A01**2)
c0A = np.sqrt((1-D01)/2)
c0B = np.sqrt((1+D01)/2)
Nc = 1000
vC = np.linspace(0,1,Nc+1)
vM = vC * np.sqrt(1-vC**2)
#
#   Plot Functions
#
#   Print Midterm Results
def Print_Mid (Rd,fSol,RJ,wJP, Ad_exct,AJ_exct,Ad_apx,AJ_apx):
    print ("")
    print ("    Mid Results")
    print ("--------------------")
    print (f"  Rd = {Rd:.1f} dBV , f_- = {fSol[0]:.5f} GHz, f_+ = {fSol[1]:.5f} GHz")
    print (f"  RJ = {RJ:.1f} dBV , f_Jp = {(wJp/(2e9*np.pi)):.5f} GHz")
    print (f" Exact : Ad = {np.abs(Ad_exct):.5f} < {np.degrees(np.angle(Ad_exct)):.2f}\N{DEGREE SIGN}  ,"
           f"  AJ = {np.abs(AJ_exct):.5f} < {np.degrees(np.angle(AJ_exct)):.2f}\N{DEGREE SIGN}")
    print (f" Approx. : Ad = {np.abs(Ad_apx):.5f} < {np.degrees(np.angle(Ad_apx)):.2f}\N{DEGREE SIGN}  ,"
           f"  AJ = {np.abs(AJ_apx):.5f} < {np.degrees(np.angle(AJ_apx)):.2f}\N{DEGREE SIGN}")
    print (f" Fit : Ad = {np.abs(Ad_fit):.5f} < {np.degrees(np.angle(Ad_fit)):.2f}\N{DEGREE SIGN}  ,"
           f"  AJ = {np.abs(AJ_fit):.5f} < {np.degrees(np.angle(AJ_fit)):.2f}\N{DEGREE SIGN}")
    print("")

#   Graph Parameters
def Write_Params() :
    Clist = [('Circuit', 0.0, '', ''), ('$F_J$',fJ,'%5.1F','Hz'), ('$E_J/E_c$',EjEc, '%4.1f',''),
             ('$L_d$', Ld, '%5.0F', 'H'), ('$Z_0$', Z0, '%5.0f', '$\\Omega$')]
    Titles(Clist, 1, 1, Ndiv=20)
    sC1 = ('%5.3f' % np.abs(c1)) + ' $\\angle$' + ('%4.1f' % phi1D) + '$^\\circ$'
    Qlist = [('Init. State', 0.0, '', ''), ('$c_0$',c0,'%5.3f',''), ('$c_1$',sC1,'','') ]
    Titles(Qlist, 1, 8, Ndiv=20)
    Slist = [('Signal', 0.0, '', ''), ('$f_J$',fJ,'%5.2F','Hz'), ('$\\kappa_J$',kJ/(2*np.pi), '%4.1F','Hz'),
             ('$f_d$', fD, '%5.2F', 'Hz'), ('$\\kappa_D$',kD/(2*np.pi), '%4.1F','Hz'),
             ('$V_s$', Vs, '%5.2F', 'V'),('$n_d$', nd, '%5.1F', 'ph')]
    Titles(Slist, 14, 1, Ndiv=20)
#
#   Reflected Wave Time Plot
def Plot_Time_Response(sMeas,Volt) :
    plt.figure(figsize=(8, 5.6))
    (rTime,sTime,kTime) = Num_Disp(max(Time),1)
    (rVolt,sVolt,kVolt) = Num_Disp(max(np.abs(Volt)),1)
    plt.plot(Time * kTime, Volt*kVolt)
    Glist = ['Reflected wave - '+ sMeas + ' Voltage', 't (' + sTime + 'sec)', 'Volt (' + sVolt + 'V)']
    Graph_Title(Glist)
    Write_Params()
#
#   Write Markers
def Write_Markers(Axes, iMark) :
    Ms = 9
    Fs = 12
    Xdel = 0.003
    Ydel = 1.5
    Ylen = 10
    match iMark :
        case 1 :
            delS = 1.5 * (fSol[1] - fSol[0])
            Axes.plot(fSol[0], Rd_dB, marker='*',
                      markersize=Ms,markerfacecolor='red',markeredgecolor='red')
            Axes.text(fSol[0], Rd_dB-Ydel, '$f_-$', fontsize=Fs,horizontalalignment='right',verticalalignment='top',color='red')
            Axes.plot(fSol[1], Rd_dB, marker='*',
                      markersize=Ms,markerfacecolor='red',markeredgecolor='red')
            Axes.text(fSol[1]+Xdel, Rd_dB-Ydel, '$f_+$', fontsize=Fs, horizontalalignment='left',verticalalignment='top',color='red')
            Axes.plot([fSol[0]-delS,fSol[1]],Rd_dB*np.array([1,1]),linestyle='dashed',color='red')
            Axes.text(fSol[0]-delS-Xdel,Rd_dB,'$R_d$',horizontalalignment='right',verticalalignment='center',fontsize=Fs,color='red')
            fJp = wJp/(2e9*np.pi)
            Axes.plot(fJp, RJ_dB, marker='*',
                  markersize=Ms, markerfacecolor='red', markeredgecolor='red')
            Axes.plot([fJp, fJp+delS], RJ_dB * np.array([1, 1]), linestyle='dashed', color='red')
            Axes.text(fJp+delS+Xdel, RJ_dB, '$R_J$', horizontalalignment='left', verticalalignment='center', fontsize=Fs,color='red')
            Axes.plot([fJp,fJp], [RJ_dB,RJ_dB-Ylen], linestyle='dashed', color='red')
            Axes.text(fJp, RJ_dB-Ylen-2*Ydel, '$f_{Jp}$', horizontalalignment='left', verticalalignment='center', fontsize=Fs,color='red')
        case 2 :
            Tlist = [('Curves', 0.0, '', ''), ('', 'Exact   ', '', '  Solid'), ('', 'Approx.', '', '  Dashed')]
            Titles(Tlist, 1, 12, Ndiv=20)
        case 3:
            Tlist = [('Curves', 0.0, '', ''), ('', 'Exact   ', '', '  Solid'), ('', '    Fit', '', '  Dashed')]
            Titles(Tlist, 1, 12, Ndiv=20)


#
#   Reflected Wave Spectrum Plot
def Plot_Spectrum(w,Vout,sMeas,iMark=0) :
    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    cColor = colors[names[0]]
    fig, ax1 = plt.subplots(figsize=(8, 5.6))
    Freq = w/(2*np.pi)
    Pout = 20*np.log10(np.abs(Vout))
    if fLog :
        ax1.semilogx(Freq, Pout)
        Glist = ['Reflected Wave Spectrum - '+sMeas, 'F (Hz)', 'Result']
    else :
        (rFreq, sFreq, kFreq) = Num_Disp(max(np.abs(Freq)), 1)
        ax1.plot(Freq*kFreq, Pout)
        Glist = ['Reflected Wave Spectrum - '+sMeas,'F ('+sFreq+'Hz)','Result']
    Graph_Title(Glist)
    ax1.tick_params(axis='y', labelcolor=cColor)
    ax1.set_ylabel('Magnitude  (dBV)',color=cColor)
    #
    cColor = colors[names[1]]
    ax2 = ax1.twinx()
    ax2.set_ylabel('Phase  (rad)', color=cColor, fontsize=14)
    AngT = np.unwrap(np.angle(Vout),axis=0)               #  Unwrap State Phase
    if fLog :
        ax2.semilogx(Freq, AngT, color=cColor)
    else :
        ax2.plot(Freq*kFreq, AngT, color=cColor)
    ax2.tick_params(axis='y', labelcolor=cColor)
    #
    Ticks_Equate(ax1,ax2,1)
    Write_Params()
    Write_Markers(ax1,iMark)
    plt.tight_layout()
#
#   Plot Spectrum Compare
def Plot_Compare (w,Vout1,Vout2,sMeas,iMark=0) :
    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    cColor = colors[names[0]]
    fig, ax1 = plt.subplots(figsize=(8, 5.6))
    Freq = w / (2 * np.pi)
    (rFreq, sFreq, kFreq) = Num_Disp(max(np.abs(Freq)), 1)
    Pout1 = 20 * np.log10(np.abs(Vout1))
    ax1.plot(Freq * kFreq, Pout1, color=cColor, linestyle='solid')
    Pout2 = 20 * np.log10(np.abs(Vout2))
    ax1.plot(Freq * kFreq, Pout2, color=cColor, linestyle='dashed')
    Glist = ['Reflected Wave Spectrum - ' + sMeas, 'F (' + sFreq + 'Hz)', 'Result']
    Graph_Title(Glist)
    ax1.tick_params(axis='y', labelcolor=cColor)
    ax1.set_ylabel('Magnitude  (dBV)', color=cColor)
    #
    cColor = colors[names[1]]
    ax2 = ax1.twinx()
    ax2.set_ylabel('Phase  (rad)', color=cColor, fontsize=14)
    AngT1 = np.unwrap(np.angle(Vout1), axis=0)  # Unwrap State Phase
    ax2.plot(Freq * kFreq, AngT1, color=cColor, linestyle='solid')
    AngT2 = np.unwrap(np.angle(Vout2), axis=0)  # Unwrap State Phase
    ax2.plot(Freq * kFreq, AngT2, color=cColor, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=cColor)
    #
    Ticks_Equate(ax1, ax2, 1)
    Write_Params()
    Write_Markers(ax1, iMark)
    plt.tight_layout()
#
#   General Function Plot
def Plot_Func(x, y, sTitle, sXaxis, sYaxis, fXunit=True, fYunit=True, sXunit="", sYunit="", iMark=0) :
    plt.figure(figsize=(8, 5.6))
    kX = kY = 1
    sX = sY = ""
    if fXunit :
        (rX,sX,kX) = Num_Disp(max(x),1)
    if fYunit :
        (rY,sY,kY) = Num_Disp(max(np.abs(y)),1)
    plt.plot(x * kX, y*kY)
    sXpar = ''
    if sXunit :
        sXpar = ' ('+sX+sXunit+')'
    sYpar = ''
    if sYunit :
        sYpar = ' ('+sY+sYunit+')'
    Glist = [sTitle, sXaxis+sXpar, sYaxis+sYpar]
    Graph_Title(Glist)
    Axes = plt.gca()
    Ms = 9
    Fs = 12
    Xdel = 0.025
    Ydel = 0.03
    Ylen = 10
    match iMark :
        case 1 :
            Axes.plot(c0A, A01, marker='*', markersize=Ms,markerfacecolor='red',markeredgecolor='red')
            Axes.text(c0A-5*Xdel, A01, '$|M_{01}|$', fontsize=Fs,horizontalalignment='right',verticalalignment='center',color='red')
            Axes.plot(c0B, A01, marker='*', markersize=Ms, markerfacecolor='red', markeredgecolor='red')
            Axes.plot([c0A-4.5*Xdel,c0B], [A01,A01], linestyle='dashed', color='red')
            Axes.plot([c0A,c0A], [0,A01], linestyle='dashed', color='red')
            Axes.text(c0A-Xdel, 0, '$c_{0A}$', fontsize=Fs,horizontalalignment='right',verticalalignment='baseline',color='red')
            Axes.plot([c0B,c0B], [0, A01], linestyle='dashed', color='red')
            Axes.text(c0B+Xdel, 0, '$c_{0B}$', fontsize=Fs, horizontalalignment='left', verticalalignment='baseline', color='red')
            c1A = np.sqrt(1-c0A**2)
            sC1A = ('%5.3f' % c1A) + ' $\\angle$' + ('%4.1f' % phi0) + '$^\\circ$'
            c1B = np.sqrt(1-c0B**2)
            sC1B = ('%5.3f' % c1B) + ' $\\angle$' + ('%4.1f' % phi0) + '$^\\circ$'
            Qlist = [('Solutions', 0.0, '', ''), ('$c_0$',c0A,'%5.3f',''), ('$c_1$',sC1A,'','') ]
            Titles(Qlist, 1, 8, Ndiv=20)
#
#   Print Midterm Results
Print_Mid (Rd_dB,fSol,RJ_dB,wJp, Ad_exct,AJ_exct,Ad_apx,AJ_apx)
#
#   Plot Time Response
Plot_Time_Response('Drive', VdT)
Plot_Time_Response('Transient',VjT)
Plot_Time_Response('Total Signal',VoT)
#   Plot Freq. Response
Plot_Spectrum(w,VfrqD, 'Drive')
Plot_Spectrum(w,VfrqJ, 'Transient')
Plot_Spectrum(w,VfrqT, 'Total Signal')
#   Plots for Qubit Coefficients
Plot_Spectrum(wn,VnT, 'Total Signal, NarrowBand View',1)
Plot_Compare(wn,VnT,VapxT,'Compare w. Approx.',2)
Plot_Compare(wn,VnT,VfitT,'Compare w. Fit',3)
Plot_Func(vC,vM,'State Coefficients Product','$c_0$','$c_0\\cdot|c_1|$',fXunit=False,fYunit=False, iMark=1)
plt.show()

