# load libraries
library(caret)
library(data.table)
library(randomForest)
library(xgboost)
library(e1071)
library(ipred)
library(kernlab)
library(MASS)
library(MLmetrics)
library(plyr)
library(adabag)
library(doSNOW)
library(foreach)
library(beepr)

# Get Data

trdata<-read.csv("C:/Kaggle/Leaves/train.csv")
tstdata<-read.csv("C:/Kaggle/Leaves/test.csv")

# Check for N/A
check<-sum(complete.cases(trdata))
check1<-sum(complete.cases(tstdata))
dim(trdata)
check
dim(tstdata)
check1

ids<-tstdata$id
seed<-1234

trdata$id<-NULL
tstdata$id<-NULL

PreObj<-preProcess(trdata[,2:193], method = c("nzv", "center", "scale"))
trntrans<-predict(PreObj, trdata[,2:193])
training<-as.data.table(cbind(species=trdata$species, trntrans))
testing<-predict(PreObj, tstdata)

training<-data.table(training, keep.rownames = F)
testing<-data.table(testing, keep.rownames = F)


load("C:/Kaggle/Leaves/m2.RData")

t2<-predict(m2, testing, type="prob")


