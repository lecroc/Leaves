## Kaggle Leaf Classification

library(MASS)
library(caret)
library(xgboost)
library(data.table)
library(Matrix)
library(methods)
library(vcd)
library(e1071)
library(DMwR)
library(mboost)
library(randomForest)
library(rpart)
library(beepr)
library(plyr)
library(gbm)
library(brnn)
library(kernlab)
library(ipred)
library(nnet)
library(MLmetrics)
library(doSNOW)
library(foreach)


seed<-75647
metric<-"logLoss"


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

trdata$id<-NULL
tstdata$id<-NULL


PreObj<-preProcess(trdata[,2:193], method = c("nzv", "center", "scale"))
trntrans<-predict(PreObj, trdata[,2:193])
training<-as.data.frame(cbind(species=trdata$species, trntrans))
testing<-predict(PreObj, tstdata)

training<-data.table(training)
testing<-data.table(testing)

#  Naive  Bayes Classifier
set.seed(seed)
m1<- naiveBayes(species~., training)


# Set up training control

fitControl<-trainControl(method="repeatedcv", number=10, repeats=3, classProbs=T, summaryFunction = multiClassSummary)

## initialize for parallel processing

getDoParWorkers()
registerDoSNOW(makeCluster(19, type="SOCK"))
getDoParWorkers()
getDoParName()



## xgbTree
set.seed(seed)
m2<-train(species~., data=training, method="xgbTree", trConrtol=fitControl, metric=metric, tuneLength=5)

## Random Forest
set.seed(seed)
m3<-train(species~., data=training, method="rf", trConrtol=fitControl, metric=metric, tuneLength=5)

## gbm
set.seed(seed)
gbmGrid<-expand.grid(interaction.depth=c(1,5,9), n.trees = (1:30)*50, shrinkage = c(.1, .05, .001), n.minobsinnode=c(6, 8, 10))
m4<-train(species~., data=training, method="gbm", trConrtol=fitControl, metric=metric, tuneGrid=gbmGrid)

## Calculate logloss for each model

predm1<- predict(m1, training, type='raw')
logloss1<-MultiLogLoss(y_true = training$species, y_pred = as.matrix(predm1))

predm2<- predict(m2, training, type='raw')
logloss2<-MultiLogLoss(y_true = training$species, y_pred = as.matrix(predm2))


predm3<- predict(m3, training, type='raw')
logloss3<-MultiLogLoss(y_true = training$species, y_pred = as.matrix(predm3))

predm4<- predict(m4, training, type='raw')
logloss4<-MultiLogLoss(y_true = training$species, y_pred = as.matrix(predm4))

# Display LogLoss

# Naive Bayes
logloss1

# xgbTree
logloss2

# Random Forest
logloss3

# gbm
logloss4

beep(7)




