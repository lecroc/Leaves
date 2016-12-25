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
training<-as.data.table(cbind(species=trdata$species, trntrans))
testing<-predict(PreObj, tstdata)

training<-data.table(training, keep.rownames = F)
testing<-data.table(testing, keep.rownames = F)

# Stacked model ensemble

# Set up training control

fitControl<-trainControl(method="repeatedcv", number=10, repeats=3, savePredictions = "final")

## initialize for parallel processing

library(doSNOW)
getDoParWorkers()
registerDoSNOW(makeCluster(19, type="SOCK"))
getDoParWorkers()
getDoParName()
library(foreach)

set.seed(seed)

# m1<-train(species~., data=training, method="xgbTree", metric=metric, trControl=fitControl)

# summary(m1)
# p1<-predict(m1, training)
# confusionMatrix(p1, training$species)

# p2<-predict(m1, testing, type="prob")

# p3<-predict(m1, testing)

# sub2<-as.data.frame(cbind(id=ids, p2))

# write.csv(sub1, "C:/Kaggle/Leaves/sub1.csv", row.names=F)

mlist<-c("rf", "svmRadial", "xgbTree", "gbm", "C5.0", "treebag")

models<-caretList(species~., data=training, metric=metric, trControl = fitControl, methodList = mlist)

results<-resamples(models)

summary(results)
dotplot(results)

modelCor(results)
splom(results)

beep(7)

