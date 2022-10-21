import numpy as np
# Ground truth DAG for pretended real-world
def log_transformation(params0, params1, params2, params3):
    sum = params0 * 2 + params1 - params2 + params3 + random.randint(0, 1)
    y = 1 / (1 + np.exp(-sum))
    y = 1 if y > 0.75 else 0
    return y

def get_asia():
    if np.random.binomial(n=1, p=0.01):
        return 0
    else:
        return 1

def get_tub(asia):
    if asia == 0:
        if np.random.binomial(n=1, p=0.05):
            return 0
        else:
            return 1
    elif asia == 1:
        if np.random.binomial(n=1, p=0.01):
            return 0
        else:
            return 1

def get_smoker_truth():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_lung_truth(smoke):
    if smoke == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif smoke == 1:
        if np.random.binomial(n=1, p=0.01):
            return 0
        else:
            return 1

def get_bronc_truth(smoke):
    if smoke == 0:
        if np.random.binomial(n=1, p=0.6):
            return 0
        else:
            return 1
    elif smoke == 1:
        if np.random.binomial(n=1, p=0.3):
            return 0
        else:
            return 1

def get_either_truth(lung, tub):
    if lung == 0 or tub == 0:
            return 0
    else:
        return 1

def get_xray_truth(either):
    if either == 0:
        if np.random.binomial(n=1, p=0.98):
            return 0
        else:
            return 1
    elif either == 1:
        if np.random.binomial(n=1, p=0.05):
            return 0
        else:
            return 1

def get_dyspnoea_truth(bronc, either):
    if bronc == 0 and either == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif bronc == 1 and either == 0:
        if np.random.binomial(n=1, p=0.7):
            return 0
        else:
            return 1
    elif bronc == 0 and either == 1:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif bronc == 1 and either == 1:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1

def get_AppOK():
    if np.random.binomial(n=1, p=0.995):
        return 0
    else:
        return 1

def get_DataFile():
    if np.random.binomial(n=1, p=0.995):
        return 0
    else:
        return 1

def get_DskLocal():
    if np.random.binomial(n=1, p=0.97):
        return 0
    else:
        return 1

def get_PrtSpool():
    if np.random.binomial(n=1, p=0.95):
        return 0
    else:
        return 1

def get_PrtOn():
    if np.random.binomial(n=1, p=0.9):
        return 0
    else:
        return 1

def get_PrtPaper():
    if np.random.binomial(n=1, p=0.98):
        return 0
    else:
        return 1

def get_NetPrint():
    if np.random.binomial(n=1, p=0.8):
        return 0
    else:
        return 1

def get_PrtDriver():
    if np.random.binomial(n=1, p=0.9):
        return 0
    else:
        return 1

def get_PrtThread():
    if np.random.binomial(n=1, p=0.9999):
        return 0
    else:
        return 1

def get_DrvSet():
    if np.random.binomial(n=1, p=0.99):
        return 0
    else:
        return 1

def get_DrvOK():
    if np.random.binomial(n=1, p=0.99):
        return 0
    else:
        return 1

def get_PrtSel():
    if np.random.binomial(n=1, p=0.99):
        return 0
    else:
        return 1

def get_PrtPath():
    if np.random.binomial(n=1, p=0.97):
        return 0
    else:
        return 1

def get_NtwrkCnfg():
    if np.random.binomial(n=1, p=0.98):
        return 0
    else:
        return 1

def get_PTROFFLINE():
    if np.random.binomial(n=1, p=0.7):
        return 0
    else:
        return 1

def get_PrtCbl():
    if np.random.binomial(n=1, p=0.98):
        return 0
    else:
        return 1

def get_PrtPort():
    if np.random.binomial(n=1, p=0.99):
        return 0
    else:
        return 1

def get_CblPrtHrdwrOK():
    if np.random.binomial(n=1, p=0.99):
        return 0
    else:
        return 1

def get_DSApplctn():
    if np.random.binomial(n=1, p=0.15):
        return 0
    else:
        return 1

def get_PrtMpTPth():
    if np.random.binomial(n=1, p=0.8):
        return 0
    else:
        return 1

def get_PrtMem():
    if np.random.binomial(n=1, p=0.95):
        return 0
    else:
        return 1

def get_PrtTimeOut():
    if np.random.binomial(n=1, p=0.94):
        return 0
    else:
        return 1

def get_FllCrrptdBffr():
    if np.random.binomial(n=1, p=0.85):
        return 0
    else:
        return 1

def get_TnrSpply():
    if np.random.binomial(n=1, p=0.995):
        return 0
    else:
        return 1



def get_PgOrnttnOK():
    if np.random.binomial(n=1, p=0.95):
        return 0
    else:
        return 1

def get_PrntngArOK():
    if np.random.binomial(n=1, p=0.98):
        return 0
    else:
        return 1

def get_ScrnFntNtPrntrFnt():
    if np.random.binomial(n=1, p=0.95):
        return 0
    else:
        return 1

def get_GrphcsRltdDrvrSttngs():
    if np.random.binomial(n=1, p=0.95):
        return 0
    else:
        return 1

def get_EPSGrphc():
    if np.random.binomial(n=1, p=0.99):
        return 0
    else:
        return 1

def get_PrtPScript():
    if np.random.binomial(n=1, p=0.4):
        return 0
    else:
        return 1

def get_TrTypFnts():
    if np.random.binomial(n=1, p=0.9):
        return 0
    else:
        return 1

def get_FntInstlltn():
    if np.random.binomial(n=1, p=0.98):
        return 0
    else:
        return 1

def get_PrntrAccptsTrtyp():
    if np.random.binomial(n=1, p=0.9):
        return 0
    else:
        return 1

def get_PrtQueue():
    if np.random.binomial(n=1, p=0.99):
        return 0
    else:
        return 1

def get_AppData(AppOK, DataFile):
    if AppOK == 0 and DataFile == 0:
        if np.random.binomial(n=1, p=0.9999):
            return 0
        else:
            return 1
    elif AppOK == 1 and DataFile == 0:
        return 1
    elif AppOK == 0 and DataFile == 1:
        return 1
    elif AppOK == 1 and DataFile == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_EMFOK(AppData, DskLocal, PrtThread):
    if AppData == 0 and DskLocal == 0 and PrtThread == 0:
        if np.random.binomial(n=1, p=0.99):
            return 0
        else:
            return 1
    elif AppData == 1 and DskLocal == 0 and PrtThread == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif AppData == 0 and DskLocal == 1 and PrtThread == 0:
            return 1
    elif AppData == 1 and DskLocal == 1 and PrtThread == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and DskLocal == 0 and PrtThread == 1:
        if np.random.binomial(n=1, p=0.05):
            return 0
        else:
            return 1
    elif AppData == 1 and DskLocal == 0 and PrtThread == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and DskLocal == 1 and PrtThread == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and DskLocal == 1 and PrtThread == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_GDIIN(AppData, PrtSpool, EMFOK):
    if AppData == 0 and PrtSpool == 0 and EMFOK == 0:
        return 0
    elif AppData == 1 and PrtSpool == 0 and EMFOK == 0:
        return 1
    elif AppData == 0 and PrtSpool == 1 and EMFOK == 0:
        return 0
    elif AppData == 1 and PrtSpool == 1 and EMFOK == 0:
        return 1
    elif AppData == 0 and PrtSpool == 0 and EMFOK == 1:
        return 1
    elif AppData == 1 and PrtSpool == 0 and EMFOK == 1:
        return 1
    elif AppData == 0 and PrtSpool == 1 and EMFOK == 1:
        return 0
    elif AppData == 1 and PrtSpool == 1 and EMFOK == 1:
        return 1

def get_GDIOUT(PrtDriver, GDIIN, DrvSet, DrvOK):
    if PrtDriver == 0 and GDIIN == 0 and DrvSet == 0 and DrvOK == 0:
        if np.random.binomial(n=1, p=0.99):
            return 0
        else:
            return 1
    elif PrtDriver == 1 and GDIIN == 0 and DrvSet == 0 and DrvOK == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif PrtDriver == 0 and GDIIN == 1 and DrvSet == 0 and DrvOK == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif PrtDriver == 1 and GDIIN == 1 and DrvSet == 0 and DrvOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtDriver == 0 and GDIIN == 0 and DrvSet == 1 and DrvOK == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif PrtDriver == 1 and GDIIN == 0 and DrvSet == 1 and DrvOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtDriver == 0 and GDIIN == 1 and DrvSet == 1 and DrvOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtDriver == 1 and GDIIN == 1 and DrvSet == 1 and DrvOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtDriver == 0 and GDIIN == 0 and DrvSet == 0 and DrvOK == 1:
        if np.random.binomial(n=1, p=0.2):
            return 0
        else:
            return 1
    elif PrtDriver == 1 and GDIIN == 0 and DrvSet == 0 and DrvOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtDriver == 0 and GDIIN == 1 and DrvSet == 0 and DrvOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtDriver == 1 and GDIIN == 1 and DrvSet == 0 and DrvOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtDriver == 0 and GDIIN == 0 and DrvSet == 1 and DrvOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtDriver == 1 and GDIIN == 0 and DrvSet == 1 and DrvOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtDriver == 0 and GDIIN == 1 and DrvSet == 1 and DrvOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtDriver == 1 and GDIIN == 1 and DrvSet == 1 and DrvOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_PrtDataOut(GDIOUT, PrtSel):
    if GDIOUT == 0 and PrtSel == 0:
        if np.random.binomial(n=1, p=0.99):
            return 0
        else:
            return 1
    elif GDIOUT == 1 and PrtSel == 0:
        return 1
    elif GDIOUT == 0 and PrtSel == 1:
        return 1
    elif GDIOUT == 1 and PrtSel == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_NetOK(PrtPath, NtwrkCnfg, PTROFFLINE):
    if PrtPath == 0 and NtwrkCnfg == 0 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.99):
            return 0
        else:
            return 1
    elif PrtPath == 1 and NtwrkCnfg == 0 and PTROFFLINE == 0:
        return 1
    elif PrtPath == 0 and NtwrkCnfg == 1 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif PrtPath == 1 and NtwrkCnfg == 1 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtPath == 0 and NtwrkCnfg == 0 and PTROFFLINE == 1:
        return 1
    elif PrtPath == 1 and NtwrkCnfg == 0 and PTROFFLINE == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtPath == 0 and NtwrkCnfg == 1 and PTROFFLINE == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtPath == 1 and NtwrkCnfg == 1 and PTROFFLINE == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_LclOK(PrtCbl, PrtPort, CblPrtHrdwrOK):
    if PrtCbl == 0 and PrtPort == 0 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.999):
            return 0
        else:
            return 1
    elif PrtCbl == 1 and PrtPort == 0 and CblPrtHrdwrOK == 0:
        return 1
    elif PrtCbl == 0 and PrtPort == 1 and CblPrtHrdwrOK == 0:
        return 1
    elif PrtCbl == 1 and PrtPort == 1 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtCbl == 0 and PrtPort == 0 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.01):
            return 0
        else:
            return 1
    elif PrtCbl == 1 and PrtPort == 0 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtCbl == 0 and PrtPort == 1 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtCbl == 1 and PrtPort == 1 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_DS_NTOK(AppData, PrtPath, PrtMpTPth, NtwrkCnfg, PTROFFLINE):
    if AppData == 0 and PrtPath == 0 and PrtMpTPth == 0 and NtwrkCnfg == 0 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.99):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtPath == 0 and PrtMpTPth == 0 and NtwrkCnfg == 0 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.2):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtPath == 1 and PrtMpTPth == 0 and NtwrkCnfg == 0 and PTROFFLINE == 0:
        return 1
    elif AppData == 1 and PrtPath == 1 and PrtMpTPth == 0 and NtwrkCnfg == 0 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtPath == 0 and PrtMpTPth == 1 and NtwrkCnfg == 0 and PTROFFLINE == 0:
        return 1
    elif AppData == 1 and PrtPath == 0 and PrtMpTPth == 1 and NtwrkCnfg == 0 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtPath == 1 and PrtMpTPth == 1 and NtwrkCnfg == 0 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtPath == 1 and PrtMpTPth == 1 and NtwrkCnfg == 0 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtPath == 0 and PrtMpTPth == 0 and NtwrkCnfg == 1 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtPath == 0 and PrtMpTPth == 0 and NtwrkCnfg == 1 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtPath == 0 and PrtMpTPth == 0 and NtwrkCnfg == 0 and PTROFFLINE == 1:
        return 1
    else:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_DS_LCLOK(AppData, PrtCbl, PrtPort, CblPrtHrdwrOK):
    if AppData == 0 and PrtCbl == 0 and PrtPort == 0 and CblPrtHrdwrOK == 0:
        return 0
    elif AppData == 1 and PrtCbl == 0 and PrtPort == 0 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtCbl == 1 and PrtPort == 0 and CblPrtHrdwrOK == 0:
        return 1
    elif AppData == 1 and PrtCbl == 1 and PrtPort == 0 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtCbl == 0 and PrtPort == 1 and CblPrtHrdwrOK == 0:
        return 1
    elif AppData == 1 and PrtCbl == 0 and PrtPort == 1 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtCbl == 1 and PrtPort == 1 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtCbl == 1 and PrtPort == 1 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtCbl == 0 and PrtPort == 0 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_PC2PRT(NetPrint, PrtDataOut, NetOK, LclOK, DSApplctn, DS_NTOK, DS_LCLOK):
    if NetPrint == 0 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 1 and NetOK == 0 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 1 and NetOK == 0 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 1 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 1 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 1 and NetOK == 1 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 1 and NetOK == 1 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 0 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 0 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 1 and NetOK == 0 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 1 and NetOK == 0 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 1 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 1 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 1 and NetOK == 1 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 1 and NetOK == 1 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 1 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 0 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 1 and NetOK == 0 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 1 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 1 and NetOK == 1 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 0 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 1 and NetOK == 0 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 1 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 1 and NetOK == 1 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 1 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 0 and LclOK == 1 and DSApplctn == 1 and DS_NTOK == 1 and DS_LCLOK == 0:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 1 and PrtDataOut == 1 and NetOK == 0 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 1 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 1 and PrtDataOut == 1 and NetOK == 1 and LclOK == 0 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 0 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 1 and PrtDataOut == 1 and NetOK == 0 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 1 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 1 and PrtDataOut == 1 and NetOK == 1 and LclOK == 1 and DSApplctn == 0 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 1 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 0 and LclOK == 1 and DSApplctn == 1 and DS_NTOK == 0 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 1 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 1 and PrtDataOut == 0 and NetOK == 0 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 1 and DS_LCLOK == 1:
        return 0
    elif NetPrint == 0 and PrtDataOut == 0 and NetOK == 1 and LclOK == 0 and DSApplctn == 1 and DS_NTOK == 1 and DS_LCLOK == 1:
        return 0
    else:
        return 1

def get_PrtData(PrtOn, PrtPaper, PC2PRT, PrtMem, PrtTimeOut, FllCrrptdBffr, TnrSpply):
    if PrtOn == 0 and PrtPaper == 0 and PC2PRT == 0 and PrtMem == 0 and PrtTimeOut == 0 and FllCrrptdBffr == 0 and TnrSpply == 0:
        if np.random.binomial(n=1, p=0.99):
            return 0
        else:
            return 1
    elif PrtOn == 1 and PrtPaper == 0 and PC2PRT == 0 and PrtMem == 0 and PrtTimeOut == 0 and FllCrrptdBffr == 0 and TnrSpply == 0:
        return 1
    elif PrtOn == 0 and PrtPaper == 1 and PC2PRT == 0 and PrtMem == 0 and PrtTimeOut == 0 and FllCrrptdBffr == 0 and TnrSpply == 0:
        return 1
    elif PrtOn == 0 and PrtPaper == 0 and PC2PRT == 1 and PrtMem == 0 and PrtTimeOut == 0 and FllCrrptdBffr == 0 and TnrSpply == 0:
        return 1
    elif PrtOn == 0 and PrtPaper == 0 and PC2PRT == 0 and PrtMem == 1 and PrtTimeOut == 0 and FllCrrptdBffr == 0 and TnrSpply == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif PrtOn == 0 and PrtPaper == 0 and PC2PRT == 0 and PrtMem == 0 and PrtTimeOut == 1 and FllCrrptdBffr == 0 and TnrSpply == 0:
        return 1
    elif PrtOn == 0 and PrtPaper == 0 and PC2PRT == 0 and PrtMem == 0 and PrtTimeOut == 0 and FllCrrptdBffr == 1 and TnrSpply == 0:
        if np.random.binomial(n=1, p=0.02):
            return 0
        else:
            return 1
    elif PrtOn == 0 and PrtPaper == 0 and PC2PRT == 0 and PrtMem == 0 and PrtTimeOut == 0 and FllCrrptdBffr == 0 and TnrSpply == 1:
        if np.random.binomial(n=1, p=0.01):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_Problem1(PrtData):
    if PrtData == 0:
        return 0
    else:
        return 1

def get_AppDtGnTm(PrtSpool):
    if PrtSpool == 0:
        return 0
    else:
        if np.random.binomial(n=1, p=0.99000001):
            return 0
        else:
            return 1

def get_PrntPrcssTm(PrtSpool):
    if PrtSpool == 0:
        if np.random.binomial(n=1, p=0.99000001):
            return 0
        else:
            return 1
    else:
        return 0

def get_DeskPrntSpd(PrtMem, AppDtGnTm, PrntPrcssTm):
    if PrtMem == 0 and AppDtGnTm == 0 and PrntPrcssTm == 0:
        if np.random.binomial(n=1, p=0.99900001):
            return 0
        else:
            return 1
    elif PrtMem == 1 and AppDtGnTm == 0 and PrntPrcssTm == 0:
        if np.random.binomial(n=1, p=0.25):
            return 0
        else:
            return 1
    elif PrtMem == 0 and AppDtGnTm == 1 and PrntPrcssTm == 0:
        if np.random.binomial(n=1, p=0.00099999):
            return 0
        else:
            return 1
    elif PrtMem == 1 and AppDtGnTm == 1 and PrntPrcssTm == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and AppDtGnTm == 0 and PrntPrcssTm == 1:
        if np.random.binomial(n=1, p=0.00099999):
            return 0
        else:
            return 1
    elif PrtMem == 1 and AppDtGnTm == 0 and PrntPrcssTm == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and AppDtGnTm == 1 and PrntPrcssTm == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 1 and AppDtGnTm == 1 and PrntPrcssTm == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_CmpltPgPrntd(PrtMem, PgOrnttnOK, PrntngArOK):
    if PrtMem == 0 and PgOrnttnOK == 0 and PrntngArOK == 0:
        if np.random.binomial(n=1, p=0.99):
            return 0
        else:
            return 1
    elif PrtMem == 1 and PgOrnttnOK == 0 and PrntngArOK == 0:
        if np.random.binomial(n=1, p=0.3):
            return 0
        else:
            return 1
    elif PrtMem == 0 and PgOrnttnOK == 1 and PrntngArOK == 0:
        if np.random.binomial(n=1, p=0.00999999):
            return 0
        else:
            return 1
    elif PrtMem == 1 and PgOrnttnOK == 1 and PrntngArOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and PgOrnttnOK == 0 and PrntngArOK == 1:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif PrtMem == 1 and PgOrnttnOK == 0 and PrntngArOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and PgOrnttnOK == 1 and PrntngArOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 1 and PgOrnttnOK == 1 and PrntngArOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_NnPSGrphc(PrtMem, GrphcsRltdDrvrSttngs, EPSGrphc):
    if PrtMem == 0 and GrphcsRltdDrvrSttngs == 0 and EPSGrphc == 0:
        if np.random.binomial(n=1, p=0.999):
            return 0
        else:
            return 1
    elif PrtMem == 1 and GrphcsRltdDrvrSttngs == 0 and EPSGrphc == 0:
        if np.random.binomial(n=1, p=0.25):
            return 0
        else:
            return 1
    elif PrtMem == 0 and GrphcsRltdDrvrSttngs == 1 and EPSGrphc == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif PrtMem == 1 and GrphcsRltdDrvrSttngs == 1 and EPSGrphc == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and GrphcsRltdDrvrSttngs == 0 and EPSGrphc == 1:
        return 1
    elif PrtMem == 1 and GrphcsRltdDrvrSttngs == 0 and EPSGrphc == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and GrphcsRltdDrvrSttngs == 1 and EPSGrphc == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 1 and GrphcsRltdDrvrSttngs == 1 and EPSGrphc == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_PSGRAPHIC(PrtMem, GrphcsRltdDrvrSttngs, EPSGrphc):
    if PrtMem == 0 and GrphcsRltdDrvrSttngs == 0 and EPSGrphc == 0:
        if np.random.binomial(n=1, p=0.999):
            return 0
        else:
            return 1
    elif PrtMem == 1 and GrphcsRltdDrvrSttngs == 0 and EPSGrphc == 0:
        if np.random.binomial(n=1, p=0.25):
            return 0
        else:
            return 1
    elif PrtMem == 0 and GrphcsRltdDrvrSttngs == 1 and EPSGrphc == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif PrtMem == 1 and GrphcsRltdDrvrSttngs == 1 and EPSGrphc == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and GrphcsRltdDrvrSttngs == 0 and EPSGrphc == 1:
        return 0
    elif PrtMem == 1 and GrphcsRltdDrvrSttngs == 0 and EPSGrphc == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and GrphcsRltdDrvrSttngs == 1 and EPSGrphc == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 1 and GrphcsRltdDrvrSttngs == 1 and EPSGrphc == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_Problem4(NnPSGrphc, PrtPScript, PSGRAPHIC):
    if NnPSGrphc == 0 and PrtPScript == 0 and PSGRAPHIC == 0:
        return 1
    elif NnPSGrphc == 1 and PrtPScript == 0 and PSGRAPHIC == 0:
        return 1
    elif NnPSGrphc == 0 and PrtPScript == 1 and PSGRAPHIC == 0:
        return 1
    elif NnPSGrphc == 1 and PrtPScript == 1 and PSGRAPHIC == 0:
        return 0
    elif NnPSGrphc == 0 and PrtPScript == 0 and PSGRAPHIC == 1:
        return 0
    elif NnPSGrphc == 1 and PrtPScript == 0 and PSGRAPHIC == 1:
        return 0
    elif NnPSGrphc == 0 and PrtPScript == 1 and PSGRAPHIC == 1:
        return 1
    elif NnPSGrphc == 1 and PrtPScript == 1 and PSGRAPHIC == 1:
        return 0

def get_TTOK(PrtMem, FntInstlltn, PrntrAccptsTrtyp):
    if PrtMem == 0 and FntInstlltn == 0 and PrntrAccptsTrtyp == 0:
        if np.random.binomial(n=1, p=0.99000001):
            return 0
        else:
            return 1
    elif PrtMem == 1 and FntInstlltn == 0 and PrntrAccptsTrtyp == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and FntInstlltn == 1 and PrntrAccptsTrtyp == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif PrtMem == 1 and FntInstlltn == 1 and PrntrAccptsTrtyp == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and FntInstlltn == 0 and PrntrAccptsTrtyp == 1:
        return 1
    elif PrtMem == 1 and FntInstlltn == 0 and PrntrAccptsTrtyp == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and FntInstlltn == 1 and PrntrAccptsTrtyp == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 1 and FntInstlltn == 1 and PrntrAccptsTrtyp == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_NnTTOK(PrtMem, ScrnFntNtPrntrFnt, FntInstlltn):
    if PrtMem == 0 and ScrnFntNtPrntrFnt == 0 and FntInstlltn == 0:
        if np.random.binomial(n=1, p=0.99000001):
            return 0
        else:
            return 1
    elif PrtMem == 1 and ScrnFntNtPrntrFnt == 0 and FntInstlltn == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and ScrnFntNtPrntrFnt == 1 and FntInstlltn == 0:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif PrtMem == 1 and ScrnFntNtPrntrFnt == 1 and FntInstlltn == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and ScrnFntNtPrntrFnt == 0 and FntInstlltn == 1:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif PrtMem == 1 and ScrnFntNtPrntrFnt == 0 and FntInstlltn == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 0 and ScrnFntNtPrntrFnt == 1 and FntInstlltn == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif PrtMem == 1 and ScrnFntNtPrntrFnt == 1 and FntInstlltn == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_Problem5(TrTypFnts, TTOK, NnTTOK):
    if TrTypFnts == 0 and TTOK == 0 and NnTTOK == 0:
        return 1
    elif TrTypFnts == 1 and TTOK == 0 and NnTTOK == 0:
        return 1
    elif TrTypFnts == 0 and TTOK == 1 and NnTTOK == 0:
        return 0
    elif TrTypFnts == 1 and TTOK == 1 and NnTTOK == 0:
        return 1
    elif TrTypFnts == 0 and TTOK == 0 and NnTTOK == 1:
        return 1
    elif TrTypFnts == 1 and TTOK == 0 and NnTTOK == 1:
        return 0
    elif TrTypFnts == 0 and TTOK == 1 and NnTTOK == 1:
        return 0
    elif TrTypFnts == 1 and TTOK == 1 and NnTTOK == 1:
        return 0

def get_LclGrbld(AppData, PrtDriver, PrtMem, CblPrtHrdwrOK):
    if AppData == 0 and PrtDriver == 0 and PrtMem == 0 and CblPrtHrdwrOK == 0:
        return 0
    elif AppData == 1 and PrtDriver == 0 and PrtMem == 0 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.2):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 1 and PrtMem == 0 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.4):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 1 and PrtMem == 0 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 0 and PrtMem == 1 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.2):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 0 and PrtMem == 1 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 1 and PrtMem == 1 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 1 and PrtMem == 1 and CblPrtHrdwrOK == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 0 and PrtMem == 0 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 0 and PrtMem == 0 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 1 and PrtMem == 0 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 1 and PrtMem == 0 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 0 and PrtMem == 1 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 0 and PrtMem == 1 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 1 and PrtMem == 1 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 1 and PrtMem == 1 and CblPrtHrdwrOK == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_NtGrbld(AppData, PrtDriver, PrtMem, NtwrkCnfg):
    if AppData == 0 and PrtDriver == 0 and PrtMem == 0 and NtwrkCnfg == 0:
        return 0
    elif AppData == 1 and PrtDriver == 0 and PrtMem == 0 and NtwrkCnfg == 0:
        if np.random.binomial(n=1, p=0.3):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 1 and PrtMem == 0 and NtwrkCnfg == 0:
        if np.random.binomial(n=1, p=0.4):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 1 and PrtMem == 0 and NtwrkCnfg == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 0 and PrtMem == 1 and NtwrkCnfg == 0:
        if np.random.binomial(n=1, p=0.2):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 0 and PrtMem == 1 and NtwrkCnfg == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 1 and PrtMem == 1 and NtwrkCnfg == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 1 and PrtMem == 1 and NtwrkCnfg == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 0 and PrtMem == 0 and NtwrkCnfg == 1:
        if np.random.binomial(n=1, p=0.4):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 0 and PrtMem == 0 and NtwrkCnfg == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 1 and PrtMem == 0 and NtwrkCnfg == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 1 and PrtMem == 0 and NtwrkCnfg == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 0 and PrtMem == 1 and NtwrkCnfg == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 0 and PrtMem == 1 and NtwrkCnfg == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 0 and PrtDriver == 1 and PrtMem == 1 and NtwrkCnfg == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif AppData == 1 and PrtDriver == 1 and PrtMem == 1 and NtwrkCnfg == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_GrbldOtpt(NetPrint, LclGrbld, NtGrbld):
    if NetPrint == 0 and LclGrbld == 0 and NtGrbld == 0:
        return 0
    elif NetPrint == 1 and LclGrbld == 0 and NtGrbld == 0:
        return 0
    elif NetPrint == 0 and LclGrbld == 1 and NtGrbld == 0:
        return 1
    elif NetPrint == 1 and LclGrbld == 1 and NtGrbld == 0:
        return 0
    elif NetPrint == 0 and LclGrbld == 0 and NtGrbld == 1:
        return 0
    elif NetPrint == 1 and LclGrbld == 0 and NtGrbld == 1:
        return 1
    elif NetPrint == 0 and LclGrbld == 1 and NtGrbld == 1:
        return 1
    elif NetPrint == 1 and LclGrbld == 1 and NtGrbld == 1:
        return 1

def get_HrglssDrtnAftrPrnt(AppDtGnTm):
    if AppDtGnTm == 0:
        if np.random.binomial(n=1, p=0.99):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1

def get_REPEAT(CblPrtHrdwrOK, NtwrkCnfg):
    if CblPrtHrdwrOK == 0 and NtwrkCnfg == 0:
        return 0
    elif CblPrtHrdwrOK == 1 and NtwrkCnfg == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif CblPrtHrdwrOK == 0 and NtwrkCnfg == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif CblPrtHrdwrOK == 1 and NtwrkCnfg == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_AvlblVrtlMmry(PrtPScript):
    if PrtPScript == 0:
        if np.random.binomial(n=1, p=0.98):
            return 0
        else:
            return 1
    else:
        return 0

def get_PSERRMEM(PrtPScript, AvlblVrtlMmry):
    if PrtPScript == 0 and AvlblVrtlMmry == 0:
        return 0
    elif PrtPScript == 1 and AvlblVrtlMmry == 0:
        return 0
    elif PrtPScript == 0 and AvlblVrtlMmry == 1:
        if np.random.binomial(n=1, p=0.05):
            return 0
        else:
            return 1
    elif PrtPScript == 1 and AvlblVrtlMmry == 1:
        return 0

def get_TstpsTxt(PrtPScript, AvlblVrtlMmry):
    if PrtPScript == 0 and AvlblVrtlMmry == 0:
        if np.random.binomial(n=1, p=0.99900001):
            return 0
        else:
            return 1
    elif PrtPScript == 1 and AvlblVrtlMmry == 0:
        return 0
    elif PrtPScript == 0 and AvlblVrtlMmry == 1:
        if np.random.binomial(n=1, p=0.00099999):
            return 0
        else:
            return 1
    elif PrtPScript == 1 and AvlblVrtlMmry == 1:
        return 0

def get_GrbldPS(GrbldOtpt, AvlblVrtlMmry):
    if GrbldOtpt == 0 and AvlblVrtlMmry == 0:
        return 0
    elif GrbldOtpt == 1 and AvlblVrtlMmry == 0:
        return 1
    elif GrbldOtpt == 0 and AvlblVrtlMmry == 1:
        if np.random.binomial(n=1, p=0.1):
            return 0
        else:
            return 1
    elif GrbldOtpt == 1 and AvlblVrtlMmry == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_IncmpltPS(CmpltPgPrntd, AvlblVrtlMmry):
    if CmpltPgPrntd == 0 and AvlblVrtlMmry == 0:
        return 0
    elif CmpltPgPrntd == 1 and AvlblVrtlMmry == 0:
        return 1
    elif CmpltPgPrntd == 0 and AvlblVrtlMmry == 1:
        if np.random.binomial(n=1, p=0.3):
            return 0
        else:
            return 1
    elif CmpltPgPrntd == 1 and AvlblVrtlMmry == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_PrtFile(PrtDataOut):
    if PrtDataOut == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.2):
            return 0
        else:
            return 1

def get_PrtIcon(NtwrkCnfg, PTROFFLINE):
    if NtwrkCnfg == 0 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.9999):
            return 0
        else:
            return 1
    elif NtwrkCnfg == 1 and PTROFFLINE == 0:
        if np.random.binomial(n=1, p=0.25):
            return 0
        else:
            return 1
    elif NtwrkCnfg == 0 and PTROFFLINE == 1:
        if np.random.binomial(n=1, p=0.7):
            return 0
        else:
            return 1
    elif NtwrkCnfg == 1 and PTROFFLINE == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_Problem6(GrbldOtpt, PrtPScript, GrbldPS):
    if GrbldOtpt == 0 and PrtPScript == 0 and GrbldPS == 0:
        return 0
    elif GrbldOtpt == 1 and PrtPScript == 0 and GrbldPS == 0:
        return 0
    elif GrbldOtpt == 0 and PrtPScript == 1 and GrbldPS == 0:
        return 0
    elif GrbldOtpt == 1 and PrtPScript == 1  and GrbldPS == 0:
        return 1
    elif GrbldOtpt == 0 and PrtPScript == 0 and GrbldPS == 1:
        return 1
    elif GrbldOtpt == 1 and PrtPScript == 0 and GrbldPS == 1:
        return 1
    elif GrbldOtpt == 0 and PrtPScript == 1 and GrbldPS == 1:
        return 0
    elif GrbldOtpt == 1 and PrtPScript == 1 and GrbldPS == 1:
        return 1

def get_Problem3(CmpltPgPrntd, PrtPScript, IncmpltPS):
    if CmpltPgPrntd == 0 and PrtPScript == 0 and IncmpltPS == 0:
        return 1
    elif CmpltPgPrntd == 1 and PrtPScript == 0 and IncmpltPS == 0:
        return 1
    elif CmpltPgPrntd == 0 and PrtPScript == 1 and IncmpltPS == 0:
        return 1
    elif CmpltPgPrntd == 1 and PrtPScript == 1  and IncmpltPS == 0:
        return 0
    elif CmpltPgPrntd == 0 and PrtPScript == 0 and IncmpltPS == 1:
        return 0
    elif CmpltPgPrntd == 1 and PrtPScript == 0 and IncmpltPS == 1:
        return 0
    elif CmpltPgPrntd == 0 and PrtPScript == 1 and IncmpltPS == 1:
        return 1
    elif CmpltPgPrntd == 1 and PrtPScript == 1 and IncmpltPS == 1:
        return 0

def get_NtSpd(DeskPrntSpd, NtwrkCnfg, PrtQueue):
    if DeskPrntSpd == 0 and NtwrkCnfg == 0 and PrtQueue == 0:
        if np.random.binomial(n=1, p=0.99900001):
            return 0
        else:
            return 1
    elif DeskPrntSpd == 1 and NtwrkCnfg == 0 and PrtQueue == 0:
        return 1
    elif DeskPrntSpd == 0 and NtwrkCnfg == 1 and PrtQueue == 0:
        if np.random.binomial(n=1, p=0.25):
            return 0
        else:
            return 1
    elif DeskPrntSpd == 1 and NtwrkCnfg == 1 and PrtQueue == 0:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif DeskPrntSpd == 0 and NtwrkCnfg == 0 and PrtQueue == 1:
        if np.random.binomial(n=1, p=0.25):
            return 0
        else:
            return 1
    elif DeskPrntSpd == 1 and NtwrkCnfg == 0 and PrtQueue == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif DeskPrntSpd == 0 and NtwrkCnfg == 1 and PrtQueue == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1
    elif DeskPrntSpd == 1 and NtwrkCnfg == 1 and PrtQueue == 1:
        if np.random.binomial(n=1, p=0.5):
            return 0
        else:
            return 1

def get_Problem2(NetPrint, DeskPrntSpd, NtSpd):
    if NetPrint == 0 and DeskPrntSpd == 0 and NtSpd == 0:
        return 0
    elif NetPrint == 1 and DeskPrntSpd == 0 and NtSpd == 0:
        return 0
    elif NetPrint == 0 and DeskPrntSpd == 1 and NtSpd == 0:
        return 1
    elif NetPrint == 1 and DeskPrntSpd == 1 and NtSpd == 0:
        return 0
    elif NetPrint == 0 and DeskPrntSpd == 0 and NtSpd == 1:
        return 0
    elif NetPrint == 1 and DeskPrntSpd == 0 and NtSpd == 1:
        return 1
    elif NetPrint == 0 and DeskPrntSpd == 1 and NtSpd == 1:
        return 1
    elif NetPrint == 1 and DeskPrntSpd == 1 and NtSpd == 1:
        return 1

def get_PrtStatPaper(PrtPaper):
    if PrtPaper == 0:
        if np.random.binomial(n=1, p=0.99900001):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.00099999):
            return 0
        else:
            return 1

def get_PrtStatToner(TnrSpply):
    if TnrSpply == 0:
        if np.random.binomial(n=1, p=0.99900001):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.00099999):
            return 0
        else:
            return 1

def get_PrtStatMem(PrtMem):
    if PrtMem == 0:
        if np.random.binomial(n=1, p=0.99900001):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.2):
            return 0
        else:
            return 1

def get_PrtStatOff(PrtOn):
    if PrtOn == 0:
        if np.random.binomial(n=1, p=0.99000001):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.00999999):
            return 0
        else:
            return 1




































