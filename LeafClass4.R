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

s1<-"C:/Kaggle/Leaves/Images"
filenames<-list.files(s1)

img<-load.image("C:/Kaggle/Leaves/Images/1.jpg")
imgr<-resize(img, 100, 100, 1, 1)
imblr<-isoblur(imgr, 1)
plot(imblr)

data<-matrix(nrow=length(filenames), ncol=10000)
names<-matrix(nrow=length(filenames), ncol=1)
for (i in 1:length(list.files(s1))) {
  img<-load.image(filenames[i])
  imgr<-resize(img, 100, 100, 1, 1)
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

combo<-combo[,c(10002, 1:10001)]

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
training<-training[, 3:10003]
training<-training[, c(10001, 1:10000)]

testing<-testing[, 3:10002]

PreObj<-preProcess(training[,2:10001], method = c("nzv", "center", "scale"))
trntrans<-predict(PreObj, training[,2:10001])
training<-as.data.table(cbind(species=trdata$species, trntrans))
testing<-predict(PreObj, testing)

training<-data.table(training, keep.rownames = F)
testing<-data.table(testing, keep.rownames = F)


control <- trainControl(method="cv", number=10, classProbs=TRUE, summaryFunction=mnLogLoss)

getDoParWorkers()
registerDoSNOW(makeCluster(7, type="SOCK"))
getDoParWorkers()
getDoParName()

set.seed(seed)

xgbTree <- train(species~., data=training, method="xgbTree", metric="logLoss", trControl=control)

pred<-predict(xgbTree, training)

cm<-confusionMatrix(training$species, pred)

testpred<-predict(xgbTree, testing, type="prob")

detailsub<-as.data.frame(cbind(id=ids, testpred))

write.csv(detailsub, "C:/Kaggle/Leaves/Leaves/detailsub.csv", row.names = F)

save(xgbTree, file="C:/Kaggle/Leaves/Leaves/xgbTree.RData")

beep(7)