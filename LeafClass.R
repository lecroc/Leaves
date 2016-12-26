## Kaggle Leaf Classification

library(MASS)
library(caret)
library(caretEnsemble)
library(xgboost)
library(data.table)
library(Matrix)
library(methods)
library(vcd)
library(e1071)
library(DMwR)
library(mboost)
library(randomForest)
library(earth)
library(rpart)
library(beepr)
library(plyr)
library(gbm)
library(survival)
library(splines)
library(bst)
library(brnn)
library(kernlab)
library(ipred)
library(nnet)
library(MLmetrics)


seed<-75647
metric<-"Accuracy"


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

# Stacked model ensemble

# Set up training control

fitControl<-trainControl(method="repeatedcv", number=10, repeats=3, savePredictions = "final", index=createFolds(training$species, 10))

## initialize for parallel processing

library(doSNOW)
getDoParWorkers()
registerDoSNOW(makeCluster(19, type="SOCK"))
getDoParWorkers()
getDoParName()
library(foreach)

set.seed(seed)

mlist<-c("rf", "svmRadial", "parRF")

models<-caretList(species~., data=training, metric=metric, trControl = fitControl, methodList = mlist)

results<-resamples(models)

summary(results)
dotplot(results)

modelCor(results)
splom(results)

# stack rf model

stackrf<-caretStack(models, method="rf", metric=metric, trControl=fitControl)
print(stackrf)

pred<- predict(stackrf ,training, type='prob')
logloss1<-MultiLogLoss(y_true = training$species, y_pred = as.matrix(pred))

logloss1

NB<- naiveBayes(species~., training)
prednb<- predict(NB, training, type='raw')
logloss2<-MultiLogLoss(y_true = training$species, y_pred = as.matrix(prednb))

logloss2

NBSub<-predict(NB, testing, type="raw")

NBSub<-as.data.frame(cbind(id=ids, NBSub))

write.csv(NBSub, "C:/Kaggle/Leaves/NBSub.csv", row.names = F)