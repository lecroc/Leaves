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
library(jpeg)
library(ripa)
library(ggplot2)
library(rasterImage)

# Get Data

system(command = "C:/Kaggle/Leaves/Images")
list.files(path = "C:/Kaggle/Leaves/Images")

s1<-"C:/Kaggle/Leaves/Images"
list.files(s1)

img<-readJPEG("C:/Kaggle/Leaves/Images/1.jpg")

img<-as.matrix(img)

trdata<-read.csv("C:/Kaggle/Leaves/train.csv")
tstdata<-read.csv("C:/Kaggle/Leaves/test.csv")

Ptero<-subset(trdata, species=="Pterocarya_Stenoptera")
