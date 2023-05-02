import numpy as np

def get_GOAL_2():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_3():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_4():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_5():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_6():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_7():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_DISPLACEM0():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_RApp1(DISPLACEM0, SNode_3):
    if DISPLACEM0 == 0 and SNode_3 == 0:
        if np.random.binomial(n=1, p=1):
            return 0
        else:
            return 1
    elif DISPLACEM0 == 1 and SNode_3 == 0:
        return 1
    elif DISPLACEM0 == 0 and SNode_3 == 1:
        return 1
    elif DISPLACEM0 == 1 and SNode_3 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1

def get_GIVEN_1():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_RApp2(GIVEN_1):
    if GIVEN_1 == 0:
        return 0
    elif GIVEN_1 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1

def get_SNode_8(RApp1, RApp2):
    if RApp1 == 0 and RApp2 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif RApp1 == 1 and RApp2 == 0:
        return 1
    elif RApp1 == 0 and RApp2 == 1:
        return 1
    elif RApp1 == 1 and RApp2 == 1:
        return 1

def get_SNode_9():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_10():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_11():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_12():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_13():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_14():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_15():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_16():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_17():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_18():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_SNode_19():
    if np.random.binomial(n=1, p=0.02):
        return 0
    else:
        return 1

def get_NEED1():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_20(SNode_16, NEED1):
    if SNode_16 == 0 and NEED1 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_16 == 1 and NEED1 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_16 == 0 and NEED1 == 1:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_16 == 1 and NEED1 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1

def get_GRAV2():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_21(SNode_20, GRAV2):
    if SNode_20 == 0 and GRAV2 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_20 == 1 and GRAV2 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_20 == 0 and GRAV2 == 1:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_20 == 1 and GRAV2 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1

def get_VALUE3():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_24(SNode_21, VALUE3):
    if SNode_21 == 0 and VALUE3 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_21 == 1 and VALUE3 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_21 == 0 and VALUE3 == 1:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_21 == 1 and VALUE3 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1

def get_SLIDING4():
    if np.random.binomial(n=1, p=0.1):
        return 0
    else:
        return 1

def get_SNode_25(SNode_15, SLIDING4):
    if SNode_15 == 0 and SLIDING4 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_15 == 1 and SLIDING4 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_15 == 0 and SLIDING4 == 1:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_15 == 1 and SLIDING4 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1

def get_CONSTANT5():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_26(SNode_11, CONSTANT5):
    if SNode_11 == 0 and CONSTANT5 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_11 == 1 and CONSTANT5 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_11 == 0 and CONSTANT5 == 1:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_11 == 1 and CONSTANT5 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1

def get_KNOWN6():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_VELOCITY7():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_47(SNode_3, VELOCITY7):
    if SNode_3 == 0 and VELOCITY7 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_3 == 1 and VELOCITY7 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_3 == 0 and VELOCITY7 == 1:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_3 == 1 and VELOCITY7 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1

def get_RApp3(KNOWN6, SNode_26, SNode_47):
    if KNOWN6 == 0 and SNode_26 == 0 and SNode_47 == 0:
        if np.random.binomial(n=1, p=1):
            return 0
        else:
            return 1
    elif KNOWN6 == 1 and SNode_26 == 0 and SNode_47 == 0:
        if np.random.binomial(n=1, p=1):
            return 0
        else:
            return 1
    elif KNOWN6 == 0 and SNode_26 == 1 and SNode_47 == 0:
        if np.random.binomial(n=1, p=1):
            return 0
        else:
            return 1
    elif KNOWN6 == 1 and SNode_26 == 1 and SNode_47 == 0:
        if np.random.binomial(n=1, p=1):
            return 0
        else:
            return 1
    elif KNOWN6 == 0 and SNode_26 == 0 and SNode_47 == 1:
        if np.random.binomial(n=1, p=1):
            return 0
        else:
            return 1
    elif KNOWN6 == 1 and SNode_26 == 0 and SNode_47 == 0:
        if np.random.binomial(n=1, p=1):
            return 0
        else:
            return 1
    elif KNOWN6 == 0 and SNode_26 == 1 and SNode_47 == 1:
        if np.random.binomial(n=1, p=1):
            return 0
        else:
            return 1
    elif KNOWN6 == 1 and SNode_26 == 1 and SNode_47 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1

def get_KNOWN8():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_RApp4(KNOWN8, SNode_11):
    if KNOWN8 == 0 and SNode_11 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif KNOWN8 == 1 and SNode_11 == 0:
        if np.random.binomial(n=1, p=0):
            return 0
        else:
            return 1
    elif KNOWN8 == 0 and SNode_11 == 1:
        if np.random.binomial(n=1, p=0):
            return 0
        else:
            return 1
    elif KNOWN8 == 1 and SNode_11 == 1:
        if np.random.binomial(n=1, p=0):
            return 0
        else:
            return 1

def get_SNode_27(RApp3, RApp4):
    if RApp3 == 0 and RApp4 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif RApp3 == 1 and RApp4 == 0:
        if np.random.binomial(n=1, p=0):
            return 0
        else:
            return 1
    elif RApp3 == 0 and RApp4 == 1:
        if np.random.binomial(n=1, p=0):
            return 0
        else:
            return 1
    elif RApp3 == 1 and RApp4 == 1:
        if np.random.binomial(n=1, p=0):
            return 0
        else:
            return 1

def get_COMPO16():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_48(GOAL_2, COMPO16):
    if GOAL_2 == 0 and COMPO16 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif GOAL_2 == 1 and COMPO16 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif GOAL_2 == 0 and COMPO16 == 1:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif GOAL_2 == 1 and COMPO16 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1

def get_TRY12():
    if np.random.binomial(n=1, p=0.7):
        return 0
    else:
        return 1

def get_TRY11(TRY12):
    if TRY12 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif TRY12 == 1:
        return 1

def get_GOAL_49(SNode_5, SNode_6, GOAL_48, TRY11):
    if SNode_5 == 1 and SNode_6 == 1 and GOAL_48 == 1 and TRY11 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return

def get_CHOOSE19():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_50(GOAL_49, CHOOSE19):
    if GOAL_49 == 1 and CHOOSE19 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SYSTEM18():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_51(SNode_17, GOAL_50, SYSTEM18):
    if SNode_17 == 0 and GOAL_50 == 0 and SYSTEM18 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_17 == 1 and GOAL_50 == 0 and SYSTEM18 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_17 == 0 and GOAL_50 == 1 and SYSTEM18 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_17 == 1 and GOAL_50 == 1 and SYSTEM18 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_17 == 0 and GOAL_50 == 0 and SYSTEM18 == 1:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_17 == 1 and GOAL_50 == 0 and SYSTEM18 == 1:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_17 == 0 and GOAL_50 == 1 and SYSTEM18 == 1:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_17 == 1 and GOAL_50 == 1 and SYSTEM18 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1

def get_KINEMATI17():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_52(SNode_51, KINEMATI17):
    if SNode_51 == 0 and KINEMATI17 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_51 == 1 and KINEMATI17 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_51 == 0 and KINEMATI17 == 1:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_51 == 1 and KINEMATI17 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1

def get_IDENTIFY10():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_53(GOAL_49, SNode_52, IDENTIFY10):
    if GOAL_49 == 1 and SNode_52 == 1 and IDENTIFY10 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_IDENTIFY9():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_28(SNode_27, GOAL_53, IDENTIFY9):
    if SNode_27 == 1 and GOAL_53 == 1 and IDENTIFY9 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_TRY13(TRY12):
    if TRY12 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif TRY12 == 1:
        return 1

def get_TRY14(TRY12):
    if TRY12 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif TRY12 == 1:
        return 1

def get_TRY15(TRY12):
    if TRY12 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif TRY12 == 1:
        return 1

def get_VAR20():
    if np.random.binomial(n=1, p=0.3):
        return 0
    else:
        return 1

def get_SNode_29(SNode_28, VAR20):
    if SNode_28 == 0 and VAR20 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_28 == 1 and VAR20 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_28 == 0 and VAR20 == 1:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_28 == 1 and VAR20 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1

def get_SNode_31(SNode_29, VALUE3):
    if SNode_29 == 0 and VALUE3 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_29 == 1 and VALUE3 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_29 == 0 and VALUE3 == 1:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_29 == 1 and VALUE3 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1

def get_GIVEN21():
    if np.random.binomial(n=1, p=0.1):
        return 0
    else:
        return 1

def get_SNode_33(SNode_10, GIVEN21):
    if SNode_10 == 0 and GIVEN21 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_10 == 1 and GIVEN21 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_10 == 0 and GIVEN21 == 1:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif SNode_10 == 1 and GIVEN21 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1

def get_SNode_34(SNode_10, CONSTANT5):
    if SNode_10 == 0 and CONSTANT5 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_10 == 1 and CONSTANT5 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_10 == 0 and CONSTANT5 == 1:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif SNode_10 == 1 and CONSTANT5 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1

def get_VECTOR27():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_APPLY32():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_56(GOAL_49, SNode_52, APPLY32):
    if GOAL_49 == 1 and SNode_52 == 1 and APPLY32 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_CHOOSE35():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_57(GOAL_56, CHOOSE35):
    if GOAL_56 == 1 and CHOOSE35 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_MAXIMIZE34():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_59(SNode_7, GOAL_57, MAXIMIZE34):
    if SNode_7 == 1 and GOAL_57 == 1 and MAXIMIZE34 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_AXIS33():
    if np.random.binomial(n=1, p=0.2):
        return 0
    else:
        return 1

def get_SNode_60(SNode_59, AXIS33):
    if SNode_59 == 1 and AXIS33 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_WRITE31():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_61(GOAL_56, SNode_60, WRITE31):
    if GOAL_56 == 1 and SNode_60 == 1 and WRITE31 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_WRITE30():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_62(GOAL_61, WRITE30):
    if GOAL_61 == 1 and WRITE30 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_RESOLVE37():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_63(SNode_28, GOAL_62, RESOLVE37):
    if SNode_28 == 1 and GOAL_62 == 1 and RESOLVE37 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_NEED36():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_64(GOAL_63, NEED36):
    if GOAL_63 == 1 and NEED36 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_41(SNode_9, CONSTANT5):
    if SNode_9 == 1 and CONSTANT5 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_42(SNode_8, SNode_41, KNOWN6):
    if SNode_8 == 1 and SNode_41 == 1 and KNOWN6 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_IDENTIFY39():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_43(SNode_42, GOAL_53, IDENTIFY39):
    if SNode_42 == 1 and GOAL_53 == 1 and IDENTIFY39 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_RESOLVE38():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_66(SNode_43, GOAL_62, RESOLVE38):
    if SNode_43 == 1 and GOAL_62 == 1 and RESOLVE38 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_67(GOAL_66, NEED36):
    if GOAL_66 == 1 and NEED36 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_IDENTIFY41():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_54(GOAL_53, IDENTIFY41):
    if GOAL_53 == 1 and IDENTIFY41 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_RESOLVE40():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_69(SNode_54, GOAL_62, RESOLVE40):
    if SNode_54 == 1 and GOAL_62 == 1 and RESOLVE40 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_70(GOAL_69, NEED36):
    if GOAL_69 == 1 and NEED36 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_IDENTIFY43():
    if np.random.binomial(n=1, p=0.6):
        return 0
    else:
        return 1

def get_SNode_55(SNode_34, GOAL_53, IDENTIFY43):
    if SNode_34 == 1 and GOAL_53 == 1 and IDENTIFY43 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_RESOLVE42():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_72(SNode_55, GOAL_62, RESOLVE42):
    if SNode_55 == 1 and GOAL_62 == 1 and RESOLVE42 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_73(GOAL_72, NEED36):
    if GOAL_72 == 1 and NEED36 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_KINE29():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_74(GOAL_62, SNode_64, SNode_67, SNode_70, SNode_73, KINE29):
    if GOAL_62 == 1 and SNode_64 == 1 and SNode_67 == 1 and SNode_70 == 1 and SNode_73 == 1 and KINE29 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_VECTOR44():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_75(SNode_4, GOAL_72, SNode_73, VECTOR44):
    if SNode_4 == 1 and GOAL_72 == 1 and SNode_73 == 1 and VECTOR44 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_EQUATION28():
    if np.random.binomial(n=1, p=0.6):
        return 0
    else:
        return 1

def get_GOAL_79(SNode_74, SNode_75, EQUATION28):
    if SNode_74 == 1 and SNode_75 == 1 and EQUATION28 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_RApp5(VECTOR27, GOAL_79):
    if VECTOR27 == 1 and GOAL_79 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1
    else:
        return 0

def get_GOAL_80(SNode_75, EQUATION28):
    if SNode_75 == 1 and EQUATION28 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_RApp6(COMPO16, GOAL_80):
    if COMPO16 == 1 and GOAL_80 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1
    else:
        return 0

def get_GOAL_81(RApp5, RApp6):
    if RApp5 == 0 and RApp6 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    else:
        return 1

def get_TRY25():
    if np.random.binomial(n=1, p=0.1):
        return 0
    else:
        return 1

def get_TRY24(TRY25):
    if TRY25 == 0:
        if np.random.binomial(n=1, p=0.95):
            return 0
        else:
            return 1
    elif TRY25 == 1:
        return 1

def get_GOAL_83(GOAL_81, TRY24):
    if GOAL_81 == 1 and TRY24 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_CHOOSE47():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_84(GOAL_83, CHOOSE47):
    if GOAL_83 == 1 and CHOOSE47 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SYSTEM46():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_86(GOAL_84, SYSTEM46):
    if GOAL_84 == 1 and SYSTEM46 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_NEWTONS45():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_156(SNode_86, NEWTONS45):
    if SNode_86 == 1 and NEWTONS45 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_DEFINE23():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_98(GOAL_83, SNode_156, DEFINE23):
    if GOAL_83 == 1 and SNode_156 == 1 and DEFINE23 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_IDENTIFY22():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_37(GOAL_98, IDENTIFY22):
    if GOAL_98 == 1 and IDENTIFY22 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_TRY26(TRY25):
    if TRY25 == 0:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1
    elif TRY25 == 1:
        return 1

def get_SNode_38(SNode_37, VAR20):
    if SNode_37 == 1 and VAR20 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_40(SNode_38, VALUE3):
    if SNode_38 == 1 and VALUE3 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_SNode_44(SNode_43, VAR20):
    if SNode_43 == 1 and VAR20 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_46(SNode_44, VALUE3):
    if SNode_44 == 1 and VALUE3 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_NULL48():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_65(SNode_29, GOAL_63, SNode_64, NULL48):
    if SNode_29 == 1 and GOAL_63 == 1 and SNode_64 == 1 and NULL48 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_SNode_68(GOAL_66, SNode_67, VECTOR44):
    if GOAL_66 == 1 and SNode_67 == 1 and VECTOR44 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_SNode_71(GOAL_69, SNode_70, VECTOR44):
    if GOAL_69 == 1 and SNode_70 == 1 and VECTOR44 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_FIND49():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_87(GOAL_83, SNode_156, FIND49):
    if GOAL_83 == 1 and SNode_156 == 1 and FIND49 == 1:
        if np.random.binomial(n=1, p=0.00007):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.7):
            return 0
        else:
            return 1

def get_NORMAL50():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_88(SNode_25, GOAL_87, NORMAL50):
    if SNode_25 == 1 and GOAL_87 == 1 and NORMAL50 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_STRAT_90():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_NORMAL52():
    if np.random.binomial(n=1, p=0.646):
        return 0
    else:
        return 1

def get_INCLINE51(NORMAL52):
    if NORMAL52 == 0:
        return 0
    else:
        return 1

def get_SNode_91(SNode_88, SNode_12, SNode_13, STRAT_90, INCLINE51):
    if SNode_88 == 1 and SNode_12 == 1 and SNode_13 == 1 and STRAT_90 == 1 and INCLINE51 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_HORIZ53(NORMAL52):
    if NORMAL52 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    elif NORMAL52 == 1:
        return 1


def get_BUGGY54(NORMAL52):
    if NORMAL52 == 0:
        if np.random.binomial(n=1, p=0.2):
            return 0
        else:
            return 1
    elif NORMAL52 == 1:
        return 0

def get_SNode_92(SNode_12, STRAT_90, BUGGY54):
    if SNode_12 == 1 and STRAT_90 == 0 and BUGGY54 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_IDENTIFY55():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_93(GOAL_87, SNode_88, IDENTIFY55):
    if GOAL_87 == 1 and SNode_88 == 1 and IDENTIFY55 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_WEIGHT56():
    if np.random.binomial(n=1, p=0.3):
        return 0
    else:
        return 1

def get_SNode_94(SNode_16, SNode_33, GOAL_87, WEIGHT56):
    if SNode_16 == 1 and SNode_33 == 1 and GOAL_87 == 1 and WEIGHT56 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_WEIGHT57():
    if np.random.binomial(n=1, p=0.3):
        return 0
    else:
        return 1

def get_SNode_95(SNode_94, WEIGHT57):
    if SNode_94 == 1 and WEIGHT57 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_97(GOAL_87, SNode_94, IDENTIFY55):
    if GOAL_87 == 1 and SNode_94 == 1 and IDENTIFY55 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_FIND58():
    if np.random.binomial(n=1, p=0.7):
        return 0
    else:
        return 1

def get_GOAL_99(GOAL_98, FIND58):
    if GOAL_98 == 1 and FIND58 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_IDENTIFY59():
    if np.random.binomial(n=1, p=0.7):
        return 0
    else:
        return 1

def get_SNode_100(GOAL_98, IDENTIFY59):
    if GOAL_98 == 1 and IDENTIFY59 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_FORCE60():
    if np.random.binomial(n=1, p=0.2):
        return 0
    else:
        return 1

def get_SNode_102(GOAL_87, SNode_88, SNode_94, FORCE60):
    if GOAL_87 == 1 and SNode_88 == 1 and SNode_94 == 1 and FORCE60 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_APPLY61():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_103(GOAL_83, SNode_102, APPLY61):
    if GOAL_83 == 1 and SNode_102 == 1 and APPLY61 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_CHOOSE62():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_104(GOAL_103, CHOOSE62):
    if GOAL_103 == 1 and CHOOSE62 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_106(GOAL_104, MAXIMIZE34):
    if GOAL_104 == 1 and MAXIMIZE34 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_152(SNode_106, AXIS33):
    if SNode_106 == 1 and AXIS33 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_WRITE63():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_107(GOAL_103, SNode_152, WRITE63):
    if GOAL_103 == 1 and SNode_152 == 1 and WRITE63 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_WRITE64():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_108(GOAL_107, WRITE64):
    if GOAL_107 == 1 and WRITE64 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_GOAL_109(GOAL_107, WRITE64):
    if GOAL_107 == 1 and WRITE64 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_GOAL65():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_110(GOAL_109, GOAL65):
    if GOAL_109 == 1 and GOAL65 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_GOAL66():
    if np.random.binomial(n=1, p=0.6):
        return 0
    else:
        return 1

def get_GOAL_111(GOAL_109, GOAL66):
    if GOAL_109 == 1 and GOAL66 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_NEED67():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_RApp7(NEED67, GOAL_109):
    if NEED67 == 1 and GOAL_109 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1
    else:
        return 0

def get_RApp8(NEED36, GOAL_111):
    if NEED36 == 1 and GOAL_111 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1
    else:
        return 0

def get_SNode_112(RApp7, RApp8):
    if RApp7 == 0 and RApp8 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    else:
        return 1

def get_GOAL68():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_113(GOAL_110, GOAL68):
    if GOAL_110 == 1 and GOAL68 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_GOAL_114(GOAL_110, GOAL68):
    if GOAL_110 == 1 and GOAL68 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_115(GOAL_114, NEED36):
    if GOAL_114 == 1 and NEED36 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_VECTOR69():
    if np.random.binomial(n=1, p=0.7):
        return 0
    else:
        return 1

def get_SNode_116(SNode_95, SNode_97, GOAL_114, SNode_115, VECTOR69):
    if SNode_95 == 1 and SNode_97 == 1 and GOAL_114 == 1 and SNode_115 == 1 and VECTOR69 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_SNode_117(GOAL_113, NEED36):
    if GOAL_113 == 1 and NEED36 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_VECTOR70():
    if np.random.binomial(n=1, p=0.1):
        return 0
    else:
        return 1

def get_SNode_118(SNode_91, GOAL_113, VECTOR70):
    if SNode_91 == 1 and GOAL_113 == 1 and VECTOR70 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_EQUAL71():
    if np.random.binomial(n=1, p=0.4):
        return 0
    else:
        return 1

def get_SNode_119(SNode_93, SNode_117, SNode_118, EQUAL71):
    if SNode_93 == 1 and SNode_117 == 1 and SNode_118 == 1 and EQUAL71 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_SNode_120(SNode_92, SNode_93, GOAL_113, SNode_117, VECTOR69):
    if SNode_92 == 1 and SNode_93 == 1 and GOAL_113 == 1 and SNode_117 == 1 and VECTOR69 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_GOAL72():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_121(GOAL_109, GOAL72):
    if GOAL_109 == 1 and GOAL72 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_122(GOAL_121, NEED36):
    if GOAL_121 == 1 and NEED36 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_VECTOR73():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_123(SNode_4, SNode_100, GOAL_121, SNode_122, VECTOR73):
    if SNode_4 == 1 and SNode_100 == 1 and GOAL_121 == 1 and SNode_122 == 1 and VECTOR73 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_NEWTONS74():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_124(SNode_37, GOAL_109, SNode_112, SNode_122, NEWTONS74):
    if SNode_37 == 1 and GOAL_109 == 1 and SNode_112 == 1 and SNode_122 == 1 and NEWTONS74 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_SUM75():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_125(GOAL_109, SNode_112, SUM75):
    if GOAL_109 == 1 and SNode_112 == 1 and SUM75 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_GOAL_126(GOAL_108, GOAL65):
    if GOAL_108 == 1 and GOAL65 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_GOAL_127(GOAL_108, GOAL66):
    if GOAL_108 == 1 and GOAL66 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_RApp9(NEED67, GOAL_108):
    if NEED67 == 1 and GOAL_108 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1
    else:
        return 0

def get_RApp10(NEED36, GOAL_127):
    if NEED36 == 1 and GOAL_127 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1
    else:
        return 0

def get_SNode_128(RApp9, RApp10):
    if RApp9 == 0 and RApp10 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    else:
        return 1

def get_GOAL_129(GOAL_126, GOAL68):
    if GOAL_126 == 1 and GOAL68 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_GOAL_130(GOAL_126, GOAL68):
    if GOAL_126 == 1 and GOAL68 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_131(GOAL_130, NEED36):
    if GOAL_130 == 1 and NEED36 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_132(SNode_95, SNode_97, GOAL_130, SNode_131, VECTOR69):
    if SNode_95 == 1 and SNode_97 == 1 and GOAL_130 == 1 and SNode_131 == 1 and VECTOR69 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_SNode_133(GOAL_129, NEED36):
    if GOAL_129 == 1 and NEED36 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_134(SNode_91, SNode_93, GOAL_129, SNode_133, VECTOR73):
    if SNode_91 == 1 and SNode_93 == 1 and GOAL_129 == 1 and SNode_133 == 1 and VECTOR73 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_SNode_135(SNode_92, SNode_93, GOAL_129, SNode_133, VECTOR69):
    if SNode_92 == 1 and SNode_93 == 1 and GOAL_129 == 1 and SNode_133 == 1 and VECTOR69 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_SNode_154(GOAL_121, NEED36):
    if GOAL_121 == 1 and NEED36 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_136(SNode_37, GOAL_108, SNode_128, SNode_154, NEWTONS74):
    if SNode_37 == 1 and GOAL_108 == 1 and SNode_128 == 1 and SNode_154 == 1 and NEWTONS74 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_SNode_137(GOAL_108, SNode_128, SUM75):
    if GOAL_108 == 1 and SNode_128 == 1 and SUM75 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_GOAL_142(SNode_116, SNode_125, EQUATION28):
    if SNode_116 == 1 and SNode_125 == 1 and EQUATION28 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_GOAL_143(SNode_116, SNode_132, EQUATION28):
    if SNode_116 == 1 and SNode_132 == 1 and EQUATION28 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_GOAL_146(SNode_132, SNode_137, EQUATION28):
    if SNode_132 == 1 and SNode_137 == 1 and EQUATION28 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_RApp11(VECTOR27, GOAL_142):
    if VECTOR27 == 1 and GOAL_142 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1
    else:
        return 0

def get_RApp12(COMPO16, GOAL_143):
    if COMPO16 == 1 and GOAL_143 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1
    else:
        return 0

def get_RApp13(VECTOR27, GOAL_146):
    if VECTOR27 == 1 and GOAL_146 == 1:
        if np.random.binomial(n=1, p=0.0001):
            return 0
        else:
            return 1
    else:
        return 0

def get_GOAL_147(RApp11, RApp12, RApp13):
    if RApp11 == 0 and RApp12 == 0 and RApp13 == 0:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1
    else:
        return 1

def get_TRY76():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_149(GOAL_147, TRY76):
    if GOAL_147 == 1 and TRY76 == 1:
        if np.random.binomial(n=1, p=0.00007):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.7):
            return 0
        else:
            return 1

def get_APPLY77():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_GOAL_150(SNode_20, SNode_37, GOAL_149, APPLY77):
    if SNode_20 == 1 and SNode_37 == 1 and GOAL_149 == 1 and APPLY77 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_GRAV78():
    if np.random.binomial(n=1, p=0.5):
        return 0
    else:
        return 1

def get_SNode_151(GOAL_150, GRAV78):
    if GOAL_150 == 1 and GRAV78 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1

def get_GOAL_153(GOAL_108, GOAL72):
    if GOAL_108 == 1 and GOAL72 == 1:
        if np.random.binomial(n=1, p=0.00008):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.8):
            return 0
        else:
            return 1

def get_SNode_155(Node_4, SNode_100, GOAL_153, SNode_154, VECTOR44):
    if Node_4 == 1 and SNode_100 == 1 and GOAL_153 == 1 and SNode_154 == 1 and VECTOR44 == 1:
        if np.random.binomial(n=1, p=0.00009):
            return 0
        else:
            return 1
    else:
        if np.random.binomial(n=1, p=0.9):
            return 0
        else:
            return 1