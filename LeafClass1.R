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

PreObj<-preProcess(trdata[,2:193], method = c("zv", "center", "scale"))
trntrans<-predict(PreObj, trdata[,2:193])
training<-as.data.table(cbind(species=trdata$species, trntrans))
testing<-predict(PreObj, tstdata)

training<-data.table(training, keep.rownames = F)
testing<-data.table(testing, keep.rownames = F)

# prepare resampling method

control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=mnLogLoss)

getDoParWorkers()
registerDoSNOW(makeCluster(7, type="SOCK"))
getDoParWorkers()
getDoParName()

set.seed(seed)

m1 <- train(species~., data=training, method="rf", metric="logLoss", trControl=control)

m2 <- train(species~., data=training, method="xgbTree", metric="logLoss", trControl=control)

m3 <- train(species~., data=training, method="xgbLinear", metric="logLoss", trControl=control)

m4 <- train(species~., data=training, method="lda", metric="logLoss", trControl=control)

# load("C:/Kaggle/Leaves/m1.RData")
# load("C:/Kaggle/Leaves/m2.RData")
# load("C:/Kaggle/Leaves/m3.RData")
# load("C:/Kaggle/Leaves/m4.RData")

# display results

p1<-predict(m1, training, type="prob")
p2<-predict(m2, training, type="prob")
p3<-predict(m3, training, type="prob")
p4<-predict(m4, training, type="prob")

ll1<-MultiLogLoss(y_true = training$species, y_pred = as.matrix(p1))
ll2<-MultiLogLoss(y_true = training$species, y_pred = as.matrix(p2))
ll3<-MultiLogLoss(y_true = training$species, y_pred = as.matrix(p3))
ll4<-MultiLogLoss(y_true = training$species, y_pred = as.matrix(p4))


colnames(p1) <- paste("m1", colnames(p1), sep = "_")
colnames(p2) <- paste("m2", colnames(p2), sep = "_")
colnames(p3) <- paste("m3", colnames(p3), sep = "_")
colnames(p4) <- paste("m4", colnames(p4), sep = "_")

stkdata<-as.data.frame(cbind(species=training$species, p1, p2))

set.seed(seed)

m5 <- train(species~., data=stkdata, method="rf", metric="logLoss", trControl=control)

t1<-predict(m1, testing, type="prob")
t2<-predict(m2, testing, type="prob")
t3<-predict(m3, testing, type="prob")
t4<-predict(m4, testing, type="prob")

colnames(t1) <- paste("m1", colnames(t1), sep = "_")
colnames(t2) <- paste("m2", colnames(t2), sep = "_")
colnames(t3) <- paste("m3", colnames(t3), sep = "_")
colnames(t4) <- paste("m4", colnames(t4), sep = "_")

stkdatatest<-as.data.frame(cbind(t1, t2))


#  Check out different models with output manipulation

testpred<-predict(m1, testing, type="prob")  # random forest model did the best

tp<-testpred^6

for(x in seq_len(nrow(tp))){
  tp[x,] <- tp[x,]/sum(tp[x,]) 
}

# Example of probability for test no 363 before raise by 5th pwoer

tp1<-as.matrix(testpred)

barplot(tp1[363,], 
        xlab = "probability", 
        xlim = c(0,1), 
        horiz = T, 
        las=1,
        cex.names=0.3,
        main = "Before Power 5")

# Example of probability for test no 363 after raise by 5th pwoer
barplot(tp[363,], 
        xlab = "probability", 
        xlim = c(0,1), 
        horiz = T, 
        las=1,
        cex.names=0.3,
        main = "After Power 5")

stksub<-as.data.frame(cbind(id=ids, tp))



write.csv(stksub, "C:/Kaggle/Leaves/Leaves/stksub.csv", row.names = F)

save(m1, file="C:/Kaggle/Leaves/Leaves/m1.RData")
save(m2, file="C:/Kaggle/Leaves/Leaves/m2.RData")
save(m3, file="C:/Kaggle/Leaves/Leaves/m3.RData")
save(m4, file="C:/Kaggle/Leaves/Leaves/m4.RData")
save(m5, file="C:/Kaggle/Leaves/Leaves/m5.RData")

beep(7)

