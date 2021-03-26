import uproot, numpy as np
from scipy.optimize import minimize
from scipy import interpolate
from numba import jit
class wavefit(object):
    def __init__(self, tpl, length, fitlength=500):
        # self.tpl = tpl
        # modify tpl to function
        self.tpl = tpl*10#interpolate.interp1d(range(tpl.shape[0]),tpl,kind='cubic')
        self.tplMax = np.max(self.tpl)
        print(self.tplMax)
        self.tplLength = len(tpl)
        self.eid = 0
        self.wave = np.zeros(length)
        self.length = length
        self.begin = 0
        self.fitlength = fitlength
        self.fun21 = lambda x: x[1]-x[0]-5
        self.jac21 = lambda x: np.array([-1,1])
        self.fun32 = lambda x: x[2]-x[1]-50
        self.E1bound = (0.1, 400)
        self.E2bound = (0.1, 400)
        self.E3bound = (10, 400)
        self.bkgbound = (0.01,3)
        self.Tbound = (0, self.fitlength-5)
        self.chiAdd = 0
        self.options = {'eps': 0.1, 'maxiter': 5000}
        self.method = 'SLSQP'
        self.tpl2Length = 0
        self.zoombound=(0.5,1.5)
        self.error = False
        # SLSQP
    def setTpl(self, mutpl):
        # self.tpl2 = mutpl
        # modify tpl2 to function
        self.tpl2 = mutpl*10#interpolate.interp1d(range(mutpl.shape[0]),mutpl,kind='cubic')
        self.tpl2Length = len(mutpl)
        self.tpl2Max = np.max(self.tpl2)
        print(self.tpl2Max)
    def cutWave(self, eid, wave):
        self.eid = eid
        self.wave = wave
        if np.nonzero(self.wave)[0].shape[0]==0 or np.nonzero(self.wave)[0][0]>=500:
            print('error in find the nonzeros in eid:{}'.format(self.eid))
            self.begin = 500
            self.error = True
        else:
            self.begin = np.nonzero(self.wave)[0][0]
        self.hitList = self.wave[self.begin:(self.begin+self.fitlength)]
        print(self.hitList[-100:])
        tplIndex = self.hitList!=0
        self.chiAdd = np.sum(self.hitList[tplIndex]*(np.log(self.hitList[tplIndex])-1))
        self.estimatePeakPos = self.getPeakWalk(self.hitList)
        if len(self.estimatePeakPos)==1:
            # fill the array to three peak,the 12.7ns is the decay time, 3000ns is the decay time use 250 instead
            self.estimatePeakPos.append(min(self.estimatePeakPos[0]+12,self.fitlength-1))
            #self.estimatePeakPos.append(min(self.estimatePeakPos[0]+250,self.fitlength-1))
            if np.max(self.hitList[-150:])>15:
                self.estimatePeakPos.append(self.hitList.shape[0]-150+np.argmax(self.hitList[-150:]))
            else:
                self.estimatePeakPos.append(min(self.estimatePeakPos[0]+250,self.fitlength-1))
        elif len(self.estimatePeakPos)==2:
            if (self.estimatePeakPos[1]-self.estimatePeakPos[0])<10:
                self.estimatePeakPos[1] = min(self.estimatePeakPos[0]+12,self.fitlength-1)
            if np.max(self.hitList[-150:])>15:
                self.estimatePeakPos.append(self.hitList.shape[0]-150+np.argmax(self.hitList[-150:]))
            else:
                self.estimatePeakPos.append(min(self.estimatePeakPos[0]+250,self.fitlength-1))    
        else:
            if (self.estimatePeakPos[1]-self.estimatePeakPos[0])<10:
                self.estimatePeakPos[1] = min(self.estimatePeakPos[0]+12,self.fitlength-1)
            if np.max(self.hitList[-150:])>15:
                self.estimatePeakPos[2] = self.hitList.shape[0]-150+np.argmax(self.hitList[-150:])
                #print('peak is {}'.format(self.estimatePeakPos[2]))
            else:
                if (self.estimatePeakPos[2]-self.estimatePeakPos[0])<150:
                    self.estimatePeakPos[2] = min(self.estimatePeakPos[0]+250,self.fitlength-1)
        self.peakValue = self.hitList[self.estimatePeakPos]
        print(self.peakValue)
        print(self.estimatePeakPos)
        #print(self.estimatePeakPos)
        #print(self.peakValue)
        self.E1bound = (0.1, np.max([np.max(self.peakValue)/self.tplMax*1.5,10]))
        #print(self.E1bound)
        if self.tpl2Length!=0:
            self.E2bound = (0.1, np.max([np.max(self.peakValue)/self.tpl2Max*1.5,10]))
            #print(self.E2bound)
        return self.begin
    # par timepos, posvalue, bkg
    def minimizeKmu(self):
        par = [0, np.max([0,self.estimatePeakPos[1]-5]), self.peakValue[0]/self.tplMax, self.peakValue[2]/self.tpl2Max, 0.01]
        #print('fit two peak initial value: {}'.format(par))
        fitresult = minimize(self.likelihoodKmutpl, par, method=self.method, bounds=(self.Tbound, self.Tbound, self.E1bound, self.E2bound, self.bkgbound), constraints=({'type': 'ineq', 'fun':self.fun21}), options=self.options)
        return fitresult
    def minimizeKmuMI(self):
        # multi initial position
        pivotIndex = np.argmax(self.peakValue)
        p1e = np.min([np.max([self.estimatePeakPos[0]-10,0]),4])
        if self.estimatePeakPos[pivotIndex]<100:
            pivot = self.estimatePeakPos[pivotIndex]  
            p2List = np.arange(np.max([0,pivot-10]),np.min([self.fitlength-5,pivot+20]),3)
            parList = [[p1e, p2, self.peakValue[0]/self.tplMax, self.peakValue[pivotIndex]/self.tpl2Max, 0.01] for p2 in p2List]+[[np.max([0,self.estimatePeakPos[pivotIndex]-5]), self.estimatePeakPos[2]-5, self.peakValue[pivotIndex]/self.tplMax, self.peakValue[2]/self.tpl2Max, 0.01]]

        else:
            if pivotIndex == 0:
                pivotIndex = 1
            pivot = self.estimatePeakPos[pivotIndex-1]
            p2List = np.arange(np.max([0,pivot-10]),np.min([self.fitlength-5,pivot+20]),3)
            parList = [[p1e, p2, self.peakValue[0]/self.tplMax, self.peakValue[pivotIndex-1]/self.tpl2Max, 0.01] for p2 in p2List]+[[np.max([0,self.estimatePeakPos[0]-5]), self.estimatePeakPos[2]-5, self.peakValue[0]/self.tplMax, self.peakValue[2]/self.tpl2Max, 0.01]]
        fitResult = []
        for par in parList:
            # print('fit K mu initial value: {}'.format(par))
            tempResult = minimize(self.likelihoodKmutpl, par, method=self.method, bounds=(self.Tbound, self.Tbound, self.E1bound, self.E2bound, self.bkgbound), constraints=({'type': 'ineq', 'fun':self.fun21}), options=self.options)
            # print('fit K mu result: {}'.format(tempResult.x))
            if tempResult.success:
                fitResult.append(tempResult)
        return min(fitResult,key=lambda x: x.fun)
    def minimizeKpiMI(self):
        # multi initial position
        # pivot = self.estimatePeakPos[1] if self.estimatePeakPos[1]<100 else self.estimatePeakPos[0]
        # p2List = np.arange(np.max([0,pivot-10]),np.min([self.fitlength-5,pivot+10]),2)
        # parList =[[np.max([0,p2-6]), np.max([0,p2-6])+6, self.peakValue[0]/self.tplMax, self.peakValue[2]/self.tpl2Max, 0.01, 1] for p2 in p2List]
        pivotIndex = np.argmax(self.peakValue)
        p1e = np.min([np.max([self.estimatePeakPos[0]-10,0]),4])
        if self.estimatePeakPos[pivotIndex]<100:
            pivot = self.estimatePeakPos[pivotIndex]  
            p2List = np.arange(np.max([0,pivot-10]),np.min([self.fitlength-5,pivot+20]),3)
            parList = [[p1e, p2, self.peakValue[0]/self.tplMax, self.peakValue[pivotIndex]/self.tpl2Max, 0.01,1] for p2 in p2List]+[[np.max([0,self.estimatePeakPos[0]-5]), self.estimatePeakPos[2]-5, self.peakValue[0]/self.tplMax, self.peakValue[2]/self.tpl2Max, 0.01,1]]

        else:
            if pivotIndex == 0:
                pivotIndex = 1
            pivot = self.estimatePeakPos[pivotIndex-1]
            p2List = np.arange(np.max([0,pivot-10]),np.min([self.fitlength-5,pivot+10]),2)
            parList = [[p1e, p2, self.peakValue[0]/self.tplMax, self.peakValue[pivotIndex-1]/self.tpl2Max, 0.01,1] for p2 in p2List]+[[np.max([0,self.estimatePeakPos[0]-5]), self.estimatePeakPos[2]-5, self.peakValue[0]/self.tplMax, self.peakValue[2]/self.tpl2Max, 0.01,1]]

        fitResult = []
        for par in parList:
            # print('fit K pi initial value: {}'.format(par))
            tempResult = minimize(self.likelihoodKpitpl, par, method=self.method, bounds=(self.Tbound, self.Tbound, self.E1bound, self.E2bound, self.bkgbound,self.zoombound), constraints=({'type': 'ineq', 'fun':self.fun21}), options=self.options)
            # print(tempResult.x)
            if tempResult.success:
                fitResult.append(tempResult)
        return min(fitResult,key=lambda x: x.fun)
    def likelihoodKpitpl(self, paras):
        nPeak = 2
        tplLength = self.tplLength
        tpl2Length = self.tpl2Length
        expectHit = np.zeros(self.hitList.shape)
        expectHit = addTpl(expectHit, self.tpl, paras[0], paras[nPeak+0], self.tplLength, self.fitlength)
        expectHit = addTpl(expectHit, self.tpl2, paras[1], paras[nPeak+1], self.tpl2Length, self.fitlength, zoom=paras[-1],azoom=False)
        expectHit += paras[2*nPeak]
        if (expectHit==0).any():
            print(paras)
            print(expectHit)
        L = np.sum(-self.hitList*np.log(expectHit)+expectHit)
        return L
    def likelihoodKmutpl(self, paras):
        nPeak = 2
        tplLength = self.tplLength
        tpl2Length = self.tpl2Length
        expectHit = np.zeros(self.hitList.shape)
        # first 
        # if (np.int(paras[0])+tplLength)>self.fitlength:
        #     expectHit[np.int(paras[0]):self.fitlength] += self.tpl[:(self.fitlength-np.int(paras[0]))] * paras[nPeak+0]
        # else:
        #     expectHit[np.int(paras[0]):(np.int(paras[0])+tplLength)] += self.tpl * paras[nPeak+0]
        # for ti in range(np.int(paras[0])+1,np.min([self.fitlength,np.int(paras[0])+tplLength])):
        #     expectHit[ti] += self.tpl(ti-paras[0]) * paras[nPeak+0]
        expectHit = addTpl(expectHit, self.tpl, paras[0], paras[nPeak+0], self.tplLength, self.fitlength)
        # second
        # if (np.int(paras[1])+tpl2Length)>self.fitlength:
        #     expectHit[np.int(paras[1]):self.fitlength] += self.tpl2[:(self.fitlength-np.int(paras[1]))] * paras[nPeak+1]
        # else:
        #     expectHit[np.int(paras[1]):(np.int(paras[1])+tpl2Length)] += self.tpl2 * paras[nPeak+1]
        # for ti in range(np.int(paras[1])+1,np.min([self.fitlength,np.int(paras[1])+tpl2Length])):
        #     expectHit[ti] += self.tpl2(ti-paras[1]) * paras[nPeak+1]
        expectHit = addTpl(expectHit, self.tpl2, paras[1], paras[nPeak+1], self.tpl2Length, self.fitlength)
        # add the bkg
        expectHit += paras[2*nPeak]
        if (expectHit==0).any():
            print(paras)
            print(expectHit)
        L = np.sum(-self.hitList*np.log(expectHit)+expectHit)
        return L
    def fitP2Result(self,paras,azoom=True):
        nPeak = 2
        tplLength = len(self.tpl)
        tpl2Length = len(self.tpl2)
        expectHit = np.zeros(self.hitList.shape)
        # first 
        expectHit = addTpl(expectHit, self.tpl, paras[0], paras[nPeak+0], self.tplLength, self.fitlength)
        # if (np.int(paras[0])+tplLength)>self.fitlength:
        #     expectHit[np.int(paras[0]):self.fitlength] += self.tpl[:(self.fitlength-np.int(paras[0]))] * paras[nPeak+0]
        # else:
        #     expectHit[np.int(paras[0]):(np.int(paras[0])+tplLength)] += self.tpl * paras[nPeak+0]
        # second
        expectHit = addTpl(expectHit, self.tpl2, paras[1], paras[nPeak+1], self.tpl2Length, self.fitlength, zoom=paras[-1],azoom=azoom)
        # if (np.int(paras[1])+tpl2Length)>self.fitlength:
        #     expectHit[np.int(paras[1]):self.fitlength] += self.tpl2[:(self.fitlength-np.int(paras[1]))] * paras[nPeak+1]
        # else:
        #     expectHit[np.int(paras[1]):(np.int(paras[1])+tpl2Length)] += self.tpl2 * paras[nPeak+1]

        # add the bkg
        expectHit += paras[2*nPeak]
        return expectHit
    def fitP1Result(self,paras):
        nPeak = 1
        tplLength = len(self.tpl)
        expectHit = np.zeros(self.hitList.shape)
        for i in range(nPeak):
            expectHit = addTpl(expectHit, self.tpl, paras[i], paras[nPeak+i], tplLength, self.fitlength, zoom=paras[-1])
            # if (np.int(paras[i])+tplLength)>self.fitlength:
            #     expectHit[np.int(paras[i]):self.fitlength] += self.tpl[:(self.fitlength-np.int(paras[i]))] * paras[nPeak+i]
            # else:
            #     expectHit[np.int(paras[i]):(np.int(paras[i])+tplLength)] += self.tpl * paras[nPeak+i]
        # add the bkg
        expectHit += paras[2*nPeak]
        return expectHit
    def minimizeAnMI(self):
        pivotIndex = np.argmax(self.peakValue)
        p1e = np.max([self.estimatePeakPos[0]-5,0])
        p1List = np.arange(np.max([0,self.estimatePeakPos[pivotIndex]-10]),np.min([self.fitlength-5,self.estimatePeakPos[pivotIndex]+10]),2)
        parList = [[p1, self.peakValue[pivotIndex]/self.tplMax, 0.01, 1] for p1 in p1List]
        self.Tbound = (np.max([0,self.estimatePeakPos[pivotIndex]-20]), self.fitlength-5)
        fitResult = []
        failResult = []
        for par in parList:
            #print('an initial {}'.format(par))
            tempResult = minimize(self.likelihoodAn, par, method=self.method, bounds=(self.Tbound, self.E1bound, self.bkgbound,self.zoombound), args=1, options=self.options)
            #print('an result {}'.format(tempResult.x))
            if tempResult.success:
                fitResult.append(tempResult)
            else:
                failResult.append(tempResult)
        if len(fitResult)>0:
            return min(fitResult,key=lambda x: x.fun)
        else:
            print('failed fit eid: {}'.format(self.eid))
            return min(failResult,key=lambda x: x.fun)
    def minimizeP1(self):
        p1List = np.arange(np.max([0,self.estimatePeakPos[np.argmax(self.peakValue)]-10]),np.min([self.fitlength-5,self.estimatePeakPos[np.argmax(self.peakValue)]+10]),2)
        parList = [[0, p1, 0.01] for p1 in p1List]
        self.Tbound = (np.max([0,self.estimatePeakPos[np.argmax(self.peakValue)]-10]), self.fitlength-5)
        fitResult = []
        for par in parList:
            tempResult = minimize(self.likelihood, par, method=self.method, bounds=(self.Tbound, self.E1bound, self.bkgbound), args=1, options=self.options)
            if tempResult.success:
                fitResult.append(tempResult)
        return min(fitResult,key=lambda x: x.fun)
    def minimizeP2(self):
        par = [self.estimatePeakPos[0], self.estimatePeakPos[1], 20, 50, 1]
        fitresult = minimize(self.likelihood, par, method='SLSQP', bounds=(self.Tbound, self.Tbound, self.E1bound, self.E2bound, (0.01,10)), constraints=({'type': 'ineq', 'fun':self.fun21}), args=2, options={'eps': 1, 'maxiter': 5000})
        return fitresult
    def minimizeP3(self):
        par = [self.estimatePeakPos[0], self.estimatePeakPos[1], self.estimatePeakPos[2], 20, 50, 20, 1]
        fitresult = minimize(self.likelihood, par, method='SLSQP', bounds=(self.Tbound, self.Tbound, self.Tbound, self.E1bound, self.E2bound, self.E3bound, (0.01,10)), constraints=({'type': 'ineq', 'fun':self.fun21}, {'type': 'ineq', 'fun': self.fun32}), args=3, options={'eps': 1, 'maxiter': 5000})
        return fitresult
    def likelihoodAn(self, paras, *args):
        nPeak = args[0]
        tplLength = self.tplLength
        expectHit = np.zeros(self.hitList.shape)
        expectHit = addTpl(expectHit, self.tpl, paras[0], paras[nPeak+0], tplLength, self.fitlength, zoom=paras[-1])
        expectHit += paras[2*nPeak]
        if (expectHit==0).any():
            print(paras)
            print(expectHit)
        L = np.sum(-self.hitList*np.log(expectHit)+expectHit)
        return L
    def likelihood(self, paras, *args):
        nPeak = args[0]
        tplLength = self.tplLength
        expectHit = np.zeros(self.hitList.shape)
        for i in range(nPeak):
            # if (np.int(paras[i])+tplLength)>self.fitlength:
                # expectHit[np.int(paras[i]):self.fitlength] += self.tpl[:(self.fitlength-np.int(paras[i]))] * paras[nPeak+i]
            # else:
                # expectHit[np.int(paras[i]):(np.int(paras[i])+tplLength)] += self.tpl * paras[nPeak+i]
            # for ti in range(np.int(paras[i])+1,np.min([self.fitlength,np.int(paras[i])+tplLength])):
            #     expectHit[ti] += self.tpl(ti-paras[i]) * paras[nPeak+i]
            # change to jit
            expectHit = addTpl(expectHit, self.tpl, paras[i], paras[nPeak+i], tplLength, self.fitlength)
        # add the bkg
        expectHit += paras[2*nPeak]
        if (expectHit==0).any():
            print(paras)
            print(expectHit)
        L = np.sum(-self.hitList*np.log(expectHit)+expectHit)
        return L
    def getPeakWalk(self, hcount, delta=1):
        smoothCount = np.zeros(hcount.shape)
        deltaCount = np.zeros(hcount.shape)
        peakPos = []
        baseline = 6
        for i in range(delta, len(hcount)-delta):
            smoothCount[i] = np.mean(hcount[(i-delta):(i+delta+1)])
            deltaCount[i-1] = smoothCount[i] - smoothCount[i-1]
        cursorb = 0
        cursore = 0
        flag = 0
        for i in range(delta, len(hcount)-delta):
            if flag==0:
                if (deltaCount[(i-2):(i+1)]>0).all():
                    flag = 1
                    cursorb = i
            elif flag==1:
                if deltaCount[i]<-40:
                    flag = 0
                    peakPos.append(i)
                    cursore = i
                #and smoothCount[i]>(1.5*smoothCount[i-2])
                #存在一些峰会被覆盖
                elif deltaCount[i]<0 and smoothCount[i]>baseline and np.sum((deltaCount[(i+1):(i+7)]<=0)!=0)>=2 and smoothCount[i]>(np.mean(smoothCount[int(cursore*2/3+i/3):(i-2)])):
                    flag = 0
                    peakPos.append(i)
                    cursore = i
        if len(peakPos)==0:
            peakPos.append(np.argmax(hcount))
        return peakPos
    def calChi(self, likelihood):
        return 2*(likelihood + self.chiAdd)
    def calLSChi(self,expectHit):
        index = self.hitList!=0
        return np.sum((self.hitList[index]-expectHit[index])**2/self.hitList[index])
@jit(nopython=True)
def addTpl(wave, tpl, t0, A, tplLength, fitlength=500, zoom=1,azoom=True):
    end = np.int(np.ceil(t0)+tplLength*zoom)-1
    if end >fitlength:
        end = fitlength-1
    if zoom==1:
        percent = t0 - np.floor(t0)
        for ti in range(np.int(np.ceil(t0)),end):
            interpB = ti - np.int(np.ceil(t0))
            # print('interB:{};percent:{}'.format(interpB, percent))
            wave[ti] += (tpl[interpB+1]*percent +tpl[interpB]*(1-percent)) * A
    else:
        for ti in range(np.int(np.ceil(t0)),end):
            interpB = np.int(np.floor((ti - t0)/zoom))
            percent = (ti - t0)/zoom-interpB
            # print('interB:{};percent:{}'.format(interpB, percent))
            if azoom:
                wave[ti] += (tpl[interpB+1]*percent +tpl[interpB]*(1-percent)) * A/zoom
            else:
                wave[ti] += (tpl[interpB+1]*percent +tpl[interpB]*(1-percent)) * A
    return wave
