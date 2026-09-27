import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
#
#       Graph Title  :  Creates Titles and Grid for the present Graph
#
def Graph_Title (Glist, Gcolor=[0.1,0.4,0.1], FontSize=10 ) :
#
    plt.grid(True,c=Gcolor, alpha=0.33)
    plt.title(Glist[0], fontsize=FontSize+6)
    plt.xlabel(Glist[1], fontsize=FontSize+4)
    plt.ylabel(Glist[2], fontsize=FontSize+4)
    Axes = plt.gca()
    Axes.spines['bottom'].set_color(Gcolor)
    Axes.spines['top'].set_color(Gcolor)
    Axes.spines['right'].set_color(Gcolor)
    Axes.spines['left'].set_color(Gcolor)
    Axes.tick_params(axis='x', colors=Gcolor)
    Axes.tick_params(axis='y', colors=Gcolor)
#
#       Num Disp  :  Break Number to its Mantissa and Ordinate Character
#
def Num_Disp (rInp, Flag=0) :
    sTab = ['f', 'p', 'n', '$\\mu$', 'm', '', 'k', 'M', 'G', 'T']
    if rInp==0 :
        iPwr = 0
    else :
        iPwr = math.floor(math.log10(abs(rInp)))
    iPwr = math.floor(iPwr/3)
    kFactor = 10**(-3*iPwr);
    rOut = abs(rInp* kFactor)
    sUnit = sTab[iPwr+5]
    if Flag :
        return rOut, sUnit, kFactor
    else :
        return rOut, sUnit
#
#       Get Color  :  Get Color Line Vector From List
#
def Get_Color (iLine) :
    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    cColor = colors[names[iLine-1]]
    return cColor
#
#       Titles  :  Print List of Parameters on Plot
#
#   The routine gets a Cell Array with the format:
#   {'Param Name' , Param_Value, 'Print Format' , 'Param Unit'}
#   The 1'st parameter in the 1'st line is the List Title
#
def Titles (ParList, mX, mY, cColor=[0,0,0], Ndiv=25, FontSize=12 ) :
    Axes = plt.gca()
    fList=True if cColor[0]<0 else False
    #
    Xlim = Axes.get_xlim()
    fXlog = (Axes.get_xscale() == 'log')
    if not fXlog:
        DelX = (Xlim[1]-Xlim[0])/Ndiv
        Xloc = Xlim[0] + mX*DelX
    else :
        DelX = (Xlim[1]/Xlim[0])**(1/Ndiv)
        Xloc = Xlim[0] * DelX**mX
    #
    Ylim = Axes.get_ylim()
    fYlog = (Axes.get_yscale() == 'log')
    if not fYlog :
        DelY = (Ylim[1]-Ylim[0])/Ndiv
        Yloc = Ylim[1] - mY*DelY
    else :
        DelY = (Ylim[1]/Ylim[0])**(1/Ndiv)
        Yloc = Ylim[1] / DelY**mY
    Nline = len(ParList)
    if 1<Nline :
        #   List Title
        if fList:
            cColor = [0,0,0]
        Axes.text(Xloc,Yloc,ParList[0][0],fontsize=FontSize, fontweight='bold' , horizontalalignment='left', color=cColor)
        if not fXlog :
            Xloc = Xloc+DelX
        else :
            Xloc = Xloc * DelX
        #   Parameters Loop
        for iLine in range(1,Nline) :
            iFind = ParList[iLine][0].find('_')
            if iFind == -1 :
                if not fYlog :
                    Yloc = Yloc - DelY
                else :
                    Yloc = Yloc / DelY
            else :
                if not fYlog :
                    Yloc = Yloc - 1.25*DelY
                else :
                    Yloc = Yloc / DelY**1.25
            #   Prepare Line
            if ParList[iLine][2] :      #  Finite Number Case
                sEnd = ParList[iLine][2][-1]
                if math.isfinite(ParList[iLine][1]):  #  Finite Number Case
                    if sEnd!='F' :      #  Simple Number Conversion Case
                        sNum = ParList[iLine][2] % ParList[iLine][1]    #  num2str
                        sLine = sNum + ' ' + ParList[iLine][3]
                        if ParList[iLine][0]:
                            sLine = ParList[iLine][0] + '=   ' + sLine
                    else :              #  Special Number Conversion Case
                        ParList[iLine][2].replace('F','f')
                        (rNum, sUnit) = Num_Disp(ParList[iLine][1])
                        sNum = ParList[iLine][2] % rNum    #  num2str
                        sLine = sNum + ' ' + sUnit + ParList[iLine][3]
                        if ParList[iLine][0]:
                            sLine = ParList[iLine][0] + '=   ' + sLine
                else :                  #  Infinite Number Case
                    sLine = '$\\infty$' + ' ' + ParList[iLine][3]
            else :
                sLine = ParList[iLine][1] + ' ' + ParList[iLine][3]
                if ParList[iLine][0]:
                    sLine = ParList[iLine][0] + '=   ' + sLine
            #   Write Line
            if fList :
                cColor = Get_Color(iLine)
            Axes.text(Xloc, Yloc, sLine, fontsize=FontSize, horizontalalignment='left', color=cColor)


#
#       Ticks_Equate  :  Align the Right Axis Ticks with the Left Axis Ones
#
def Ticks_Equate(axL,axR, iDir=1) :
    #   Left Side
    yTL = axL.get_yticks()
    LzeroT = np.where(yTL==0.)
    Lzero = LzeroT[0]
    if Lzero.size != 0 :
        Lzero = Lzero[0]
    nTL = len(yTL)
    Dtl = yTL[1]-yTL[0]
    yLlim = axL.get_ylim()
    if yTL[0]<yLlim[0] :
        yLlow = yTL[0]
    else :
        yLlow = yTL[0] - Dtl
    #
    if yLlim[1]<yTL[-1] :
        yLhigh = yTL[-1]
    else :
        yLhigh = yTL[-1] + Dtl
    yLlim = (yLlow, yLhigh)
    axL.set_ylim(yLlim)
    axL.set_yticks(np.arange(yLlow, yLhigh+Dtl, Dtl))
    yTL = axL.get_yticks()
    nTL = len(yTL)
    #   Right Side
    yTR = axR.get_yticks()
    yRlim = (yTR[0], yTR[-1])
    axR.set_ylim(yRlim)
    yTR = axR.get_yticks()
    nTR = len(yTR)
    Dtr = (yTR[-1]-yTR[0])/(nTL-1)
    #   Equity Set
    if nTL!=nTR :
        kLR = 1 #   np.heaviside(nTL-nTR,0.5)
        Dtr = kLR * Dtr
        match iDir :
            case -1 :
                yRlow = yRlim[1] - (nTL-1) * Dtr
                yRhigh = yRlim[1]
            #
            case 0 :
                yRlow = -Lzero * Dtr
                yRhigh = yRlim[0] + (nTL-1) * Dtr
                yRhigh = yRlim[0] + (nTL-1) * Dtr
        #
            case 1 :
                yRlow = yRlim[0]
                yRhigh = yRlim[0] + (nTL - 1) * kLR * Dtr
        #
        yRlim = (yRlow, yRhigh)
        axR.set_ylim(yRlim)
        yTR_New = np.linspace(yRlow, yRhigh, nTL)
        axR.set_yticks(yTR_New)
#
#       PolySol  :  Find the Points where the Polyline Equals to a Specified Level
#
def PolySol(X,Y, Level) :
    Y0 = Y - Level
    Ylog = np.diff(Y0>=0)
    Indx = np.nonzero(Ylog)
    Xdel = np.append(np.diff(X),0)
    Ydel = np.append(np.diff(Y0), 0)
    Xsol = X[Indx]-Y0[Indx]*Xdel[Indx]/Ydel[Indx]
    return Xsol
