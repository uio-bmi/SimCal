import numpy as np
import bnlearn as bn
import pandas as pd
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import DataFrame, StrVector
from rpy2.robjects.packages import importr

def bnlearn_setup_hc(train_data, pipeline_type):
    print("this is the training data shape before structure learning", train_data.shape)
    if(pipeline_type==1 or pipeline_type==2 or pipeline_type==3):
        robjects.r('''
            library(bnlearn)
            bn_hillclimbing <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- hc(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_hillclimbing()
            ''')
    elif(pipeline_type==4):
        robjects.r('''
            library(bnlearn)
            bn_hillclimbing <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- hc(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_hillclimbing()
            ''')
    bn_hc = robjects.r['bn_hillclimbing']
    bn_train_output = bn_hc()

    result = np.array(bn_train_output)
    if(pipeline_type==1 or pipeline_type==2 or pipeline_type==3):
        result_edit = result[:, :].reshape([-1, 5])
    elif(pipeline_type==4):
        result_edit = result[:, :].reshape([-1, 11])
    print("this is the output shape after structure learning", result_edit.shape)

    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
                library(bnlearn)
                bn_hillclimbing <- function(r, verbose=FALSE) {
                databn <-read.csv("train.csv", header=FALSE)
                databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
                my_bn <- hc(databn)
                fit = bn.fit(my_bn, databn)
                training_output = rbn(my_bn, 1000, databn)
                }
                bn_hillclimbing()
                ''')
    elif (pipeline_type == 4):
        robjects.r('''
                library(bnlearn)
                bn_hillclimbing <- function(r, verbose=FALSE) {
                databn <-read.csv("train.csv", header=FALSE)
                databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
                my_bn <- hc(databn)
                fit = bn.fit(my_bn, databn)
                training_output = rbn(my_bn, 1000, databn)
                }
                bn_hillclimbing()
                ''')
    bn_hc = robjects.r['bn_hillclimbing']
    bn_train_output = bn_hc()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit_test = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit_test = result[:, :].reshape([-1, 11])

    np.savetxt('Z_est_train.csv', result_edit, delimiter=',')
    return result_edit, result_edit_test

def bnlearn_setup_tabu(train_data, pipeline_type):
    print("this is the training data shape before structure learning", train_data.shape)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_tabu <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- tabu(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_tabu()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_tabu <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- tabu(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_tabu()
            ''')
    bn_tabu = robjects.r['bn_tabu']
    bn_train_output = bn_tabu()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit = result[:, :].reshape([-1, 11])
    print("this is the output shape after structure learning", result_edit.shape)

    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_tabu <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- tabu(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_tabu()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_tabu <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- tabu(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_tabu()
            ''')
    bn_tabu = robjects.r['bn_tabu']
    bn_train_output = bn_tabu()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit_test = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit_test = result[:, :].reshape([-1, 11])
    np.savetxt('Z_est_train.csv', result_edit, delimiter=',')
    return result_edit, result_edit_test

def bnlearn_setup_iamb(train_data, pipeline_type):
    print("this is the training data shape before structure learning", train_data.shape)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_iamb <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- iamb(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_iamb()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_iamb <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- iamb(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_iamb()
            ''')
    bn_iamb = robjects.r['bn_iamb']
    bn_train_output = bn_iamb()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit = result[:, :].reshape([-1, 11])
    print("this is the output shape after structure learning", result_edit.shape)

    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_iamb <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- iamb(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_iamb()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_iamb <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- iamb(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_iamb()
            ''')
    bn_iamb = robjects.r['bn_iamb']
    bn_train_output = bn_iamb()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit_test = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit_test = result[:, :].reshape([-1, 11])
    np.savetxt('Z_est_train.csv', result_edit, delimiter=',')
    return result_edit, result_edit_test

def bnlearn_setup_pc(train_data, pipeline_type):
    print("this is the training data shape before structure learning", train_data.shape)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_pc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- pgmpy_model.stable(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_pc()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_pc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- pgmpy_model.stable(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_pc()
            ''')
    bn_pc = robjects.r['bn_pc']
    bn_train_output = bn_pc()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit = result[:, :].reshape([-1, 11])
    print("this is the output shape after structure learning", result_edit.shape)

    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_pc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- pgmpy_model.stable(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_pc()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_pc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- pgmpy_model.stable(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_pc()
            ''')
    bn_pc = robjects.r['bn_pc']
    bn_train_output = bn_pc()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit_test = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit_test = result[:, :].reshape([-1, 11])
    np.savetxt('Z_est_train.csv', result_edit, delimiter=',')
    return result_edit, result_edit_test

def bnlearn_setup_gs(train_data, pipeline_type):
    print("this is the training data shape before structure learning", train_data.shape)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_gs <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- gs(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_gs()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_gs <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- gs(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_gs()
            ''')
    bn_gs = robjects.r['bn_gs']
    bn_train_output = bn_gs()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit = result[:, :].reshape([-1, 11])
    print("this is the output shape after structure learning", result_edit.shape)

    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_gs <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- gs(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_gs()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_gs <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- gs(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_gs()
            ''')
    bn_gs = robjects.r['bn_gs']
    bn_train_output = bn_gs()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit_test = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit_test = result[:, :].reshape([-1, 11])
    np.savetxt('Z_est_train.csv', result_edit, delimiter=',')
    return result_edit, result_edit_test

def bnlearn_setup_mmhc(train_data, pipeline_type):
    print("this is the training data shape before structure learning", train_data.shape)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_mmhc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- mmhc(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_mmhc()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_mmhc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- mmhc(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_mmhc()
            ''')
    bn_mmhc = robjects.r['bn_mmhc']
    bn_train_output = bn_mmhc()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit = result[:, :].reshape([-1, 11])
    print("this is the output shape after structure learning", result_edit.shape)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_mmhc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- mmhc(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_mmhc()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_mmhc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- mmhc(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_mmhc()
            ''')
    bn_mmhc = robjects.r['bn_mmhc']
    bn_train_output = bn_mmhc()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit_test = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit_test = result[:, :].reshape([-1, 11])
    np.savetxt('Z_est_train.csv', result_edit, delimiter=',')
    return result_edit, result_edit_test

def bnlearn_setup_rsmax2(train_data, pipeline_type):
    print("this is the training data shape before structure learning", train_data.shape)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_rsmax2 <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- rsmax2(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_rsmax2()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_rsmax2 <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- rsmax2(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_rsmax2()
            ''')
    bn_rsmax2 = robjects.r['bn_rsmax2']
    bn_train_output = bn_rsmax2()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit = result[:, :].reshape([-1, 11])
    print("this is the output shape after structure learning", result_edit.shape)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_rsmax2 <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- rsmax2(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_rsmax2()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_rsmax2 <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- rsmax2(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_rsmax2()
            ''')
    bn_rsmax2 = robjects.r['bn_rsmax2']
    bn_train_output = bn_rsmax2()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit_test = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit_test = result[:, :].reshape([-1, 11])
    np.savetxt('Z_est_train.csv', result_edit_test, delimiter=',')
    return result_edit, result_edit

def bnlearn_setup_h2pc(train_data, pipeline_type):
    print("this is the training data shape before structure learning", train_data.shape)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_h2pc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- h2pc(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_h2pc()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_h2pc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- h2pc(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_h2pc()
            ''')
    bn_h2pc = robjects.r['bn_h2pc']
    bn_train_output = bn_h2pc()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit = result[:, :].reshape([-1, 11])
    print("this is the output shape after structure learning", result_edit.shape)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        robjects.r('''
            library(bnlearn)
            bn_h2pc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5")] <- lapply(databn[,c("V1","V2","V3","V4","V5")], as.factor)
            my_bn <- h2pc(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_h2pc()
            ''')
    elif (pipeline_type == 4):
        robjects.r('''
            library(bnlearn)
            bn_h2pc <- function(r, verbose=FALSE) {
            databn <-read.csv("train.csv", header=FALSE)
            databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")] <- lapply(databn[,c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")], as.factor)
            my_bn <- h2pc(databn)
            fit = bn.fit(my_bn, databn)
            training_output = rbn(my_bn, 1000, databn)
            }
            bn_h2pc()
            ''')
    bn_h2pc = robjects.r['bn_h2pc']
    bn_train_output = bn_h2pc()

    result = np.array(bn_train_output)
    if (pipeline_type == 1 or pipeline_type == 2 or pipeline_type == 3):
        result_edit_test = result[:, :].reshape([-1, 5])
    elif (pipeline_type == 4):
        result_edit_test = result[:, :].reshape([-1, 11])
    np.savetxt('Z_est_train.csv', result_edit, delimiter=',')
    return result_edit, result_edit_test
