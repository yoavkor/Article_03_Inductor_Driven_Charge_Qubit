#
#       Markov Approximation Check
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
qe = 1.602e-19
#   Circuit Parameters
fJ = 7.0e9         #  Qubit Frequency
EjEc = 50           #  Qubit's Energies Ratio
Ld = 25e-9          #  Serial Inductor
Z0 = 50             #  Cable Impedance
#   Drive Signal Parameters
sigT = 0.5e-9       #  Gaussian Input Signal Time Deviation
mSig = 5            #  Gaussian Width to Deviation Ratio
#   Integral Parameters
wJ = 2*np.pi*fJ                                 #  Qubit Freq.
Zj = hbar/(np.sqrt(2)*qe**2)/np.sqrt(EjEc)      #  Qubit Impedance
wg = 1/(2*Ld)*np.sqrt(Zj*Z0/(2*np.pi))          #  Interaction Freq.
sigW = 1/sigT                                   #  Gaussian Input Signal Freq. Deviation
delWb = mSig*sigW                               #  Gaussian Input Signal Bandwidth
wA = np.pi*(mSig*wg)**2/delWb                   #  Qubit's Response Freq.
T0 = 20/fJ          #  Integral's Peak Time
Te = 2*T0           #  Result's End Time
#
#   Calculate Result
Ntm = 1000
Time = np.linspace(0, Te, Ntm+1)
Mres = delWb * np.sinc(delWb/2*(Time-T0))
Fres = Mres/(2*np.pi)
#
#   Plot
#
#   Drive Signal
plt.figure(num=1, figsize=(8, 5.6))
(rTe,sTe,kTe) = Num_Disp(max(Time),1)
(rFres,sFres,kFres) = Num_Disp(max(Fres),1)
plt.plot(Time*kTe, Fres*kFres)
Glist = ['Complex Exponential Integral Result','t ('+sTe+'sec)','Result  ('+sFres+'Hz)']
Graph_Title(Glist)
Clist = [('Circuit',0.0,'',''), ('$F_J$',fJ,'%5.1F','Hz'), ('$\\epsilon$',EjEc,'%5.1f',''), ('$L_d$',Ld,'%4.1F','H'), \
    ('$Z_0$',Z0,'%3.0F','$\\Omega$')]
Titles (Clist, 1, 1)
Dlist = [('Signal',0.0,'',''), ('$\\sigma_t$',sigT,'%5.1F','sec'), ('m',mSig,'%5.1F','')]
Titles (Dlist, 1, 8)
Ilist = [('Integral',0.0,'',''), ('$Z_J$',Zj,'%5.1F','$\\Omega$'), ('$F_g$',wg/(2*np.pi),'%5.1F','Hz'), \
         ('$\\Delta F_B$',delWb/(2*np.pi),'%5.2F','Hz'), ('$F_A$',wA/(2*np.pi),'%5.2F','Hz')]
Titles (Ilist, 17, 1)
plt.show()
