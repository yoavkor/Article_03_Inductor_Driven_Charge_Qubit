import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
#
#       States Phase and Frequencies
#
def States_Freq_Phase (wJ,Time,AngT, rhoN=0.4) :
    TimeN = wJ*Time                     #  Reference Freq. Phase
    Ntm = np.size(AngT,0)          #  Time Vector Length
    Ns = np.size(AngT,1)           #  No. of States
    Iend = np.floor(round(rhoN*Ntm))    #  End Portion Index
    Freq_Rel = np.zeros(Ns)         #  Relative Freq. Vector
    Aphase = np.zeros((Ntm,Ns))         #  Relative Phase Matrix
    for jS in range(Ns) :               #  Loop on States
        Tend = TimeN[-Iend:]            #  Time End Portion Vector
        Aend = AngT[-Iend:,jS]          #  Angle End Portion Vector
        pfAng = np.polyfit(Tend,Aend,1) #  Linear Approx. Line Coeff.
        Freq_Rel[jS] = -pfAng[0]        #  Relative Freq.
        Aphase[:,jS] = AngT[:,jS]  + Freq_Rel[jS]*(TimeN-TimeN[0])     #  Relative State Phase
    return Freq_Rel, Aphase
#
#       Bloch Sphere Angles
#
def Bloch_Sphere_Angles (Aabs,Aphase) :
    Aelev = 2 * np.atan2(Aabs[:,1],Aabs[:,0])   #  Elevation Angle
    Aazim = Aphase[:,1] - Aphase[:,0]           #  Azimuth Angle
    return Aelev, Aazim
