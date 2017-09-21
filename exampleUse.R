library(mlbench)
source("optimizeModels.R")
set.seed(7)

# load the dataset
data(BostonHousing)

#split
validation_index <- createDataPartition(BostonHousing$medv, p=0.80, list=FALSE)
validation <- BostonHousing[-validation_index,]
training <- BostonHousing[validation_index,]

#call Function (with verbose & plot)
fmla<-formula(medv~.)
s <- findOptimalModelsReg(training,validation,fmla,verbose=T,plot=T,cores=4)

#Let's asume GBM is the best... for this example

#Obtain the best hyperparameters
print(s$gbm$bestTune)
ggplot(s$gbm$model)  

#train with 100%
model<-gbm(data = BostonHousing,fmla,n.trees = 150,interaction.depth = 3,shrinkage = 0.1,n.minobsinnode = 10)

#always save the models!!
save(model,file="GBM.RData")
