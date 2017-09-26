library(AppliedPredictiveModeling)
source("optimizeModels.R")
set.seed(7)

# load the dataset
data(abalone)

#split
validation_index <- createDataPartition(abalone$Rings, p=0.50, list=FALSE)
validation <- abalone[-validation_index,]
training <- abalone[validation_index,]

#call Function (with verbose & plot)
fmla<-formula(Rings~.)
s <- findOptimalModelsReg(training,validation,fmla,verbose=T,plot=T,cores=4)

#Let's asume GBM is the best... for this example

#Obtain the best hyperparameters
print(s$gbm$bestTune)
ggplot(s$gbm$model)  

#train with 100%
model<-gbm(data = BostonHousing,fmla,n.trees = 150,interaction.depth = 3,shrinkage = 0.1,n.minobsinnode = 10)

#always save the models!!
save(model,file="GBM.RData")
