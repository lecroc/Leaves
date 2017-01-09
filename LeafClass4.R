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
library(imager)

seed<-1234

# Get Data

system(command = "C:/Kaggle/Leaves/Images")
list.files(path = "C:/Kaggle/Leaves/Images")
setwd("C:/Kaggle/Leaves/Images")

s1<-"C:/Kaggle/Leaves/Images"
filenames<-list.files(s1)

img<-load.image("C:/Kaggle/Leaves/Images/1.jpg")
imgr<-resize(img, 12, 12, 1, 1)
imblr<-isoblur(imgr, 1)
plot(imblr)

data<-matrix(nrow=length(filenames), ncol=144)
names<-matrix(nrow=length(filenames), ncol=1)
for (i in 1:length(filenames)) {
  img<-load.image(filenames[i])
  imgr<-resize(img, 12, 12, 1, 1)
  imblr<-isoblur(imgr, 1)
  imblrv<-as.vector(imblr)
  names[i,]<-filenames[i]
  data[i,]<-imblrv}

data<-as.data.frame(data)
names<-as.data.frame(names)
names(names)<-c("Filename")

combo<-as.data.frame(cbind(names, data))
temp <- gregexpr("[0-9]+", combo$Filename)  # Numbers with any number of digits
combo$FilNumb<-as.numeric(unique(unlist(regmatches(combo$Filename, temp))))

combo<-combo[order(combo$FilNumb), ]

combo<-combo[,c(146, 1:145)]

trdata<-read.csv("C:/Kaggle/Leaves/train.csv")
tstdata<-read.csv("C:/Kaggle/Leaves/test.csv")
ids<-tstdata$id

trindex<-trdata$id
tstindex<-tstdata$id

training<-combo[is.element(combo$FilNumb, trindex), ]
testing<-combo[is.element(combo$FilNumb, tstindex), ]

dim(training)
dim(testing)

training$species<-trdata$species
training<-training[, 3:147]
training<-training[, c(145, 1:144)]

testing<-testing[, 3:146]

PreObj<-preProcess(training[,2:145], method = c("nzv", "center", "scale"))
trntrans<-predict(PreObj, training[,2:145])
training<-as.data.table(cbind(species=trdata$species, trntrans))
testing<-predict(PreObj, testing)

training<-data.table(training, keep.rownames = F)
testing<-data.table(testing, keep.rownames = F)


control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=mnLogLoss)

getDoParWorkers()
registerDoSNOW(makeCluster(7, type="SOCK"))
getDoParWorkers()
getDoParName()

set.seed(seed)

rf <- train(species~., data=training, method="rf", metric="logLoss", trControl=control, tuneLength=5)

pred<-predict(rf, training)

cm<-confusionMatrix(training$species, pred)

testpred<-predict(rf, testing, type="prob")

rf1sub<-as.data.frame(cbind(id=ids, testpred))

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

rf2sub<-as.data.frame(cbind(id=ids, tp))

write.csv(rf1sub, "C:/Kaggle/Leaves/Leaves/rf1.csv", row.names = F)
write.csv(rf2sub, "C:/Kaggle/Leaves/Leaves/rf2.csv", row.names = F)

save(rf, file="C:/Kaggle/Leaves/Leaves/rf.RData")

beep(7)