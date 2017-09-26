# load the library
library(caret)
library(doMC)
library(progress)

findOptimalModelsReg<-function(train,test,fmla, Kfolds=5, repeats=2, verbose=F, plot=F, cores=1){
  #TODO: Add center and scale as parameters:
  #preProc = c("center", "scale")
  
  # Fits a list of models to evaluate the best hyperparameter configuration
  # and expected metrics. For now, the method is repeated cross-validation.
  # 
  #
  # Args:
  #   train: training dataset. This will be internally split in different kfolds.
  #   test: test dataset. This will be used to evaluate the models.
  #   fmla: model formula. Used by the models to select target and features.
  #   Kfolds: Number of folds in the repeated crossvalidation.
  #   repeats: number of complete sets of folds to compute.
  #   verbose: If TRUE, prints summary comparison. Default is FALSE.
  #   plot: If TRUE, plots summary comparison. Default is FALSE.
  #   cores: registers the amount of cores to use for parallel computation. Default is 1 (no parallel).
  #
  # Returns:
  #   List, for each model by name, of:
  #      model: As returned by the caret train function. 
  #      bestTune: the best hyperparameter configuration found for that model.
  #      testMetrics: test metrics for each model with the test dataset.
  
  #allow multicore
  if(cores>1) registerDoMC(cores=cores)
  pb <- progress_bar$new(total = 6)
  # prepare training scheme
  control <- trainControl(method="repeatedcv",
                          number=Kfolds,
                          repeats=repeats,
                          verboseIter = FALSE,
                          returnData = FALSE,
                          allowParallel = TRUE)
  
  ### Some models requires dummy vars
  
  #This transformation SHOULD work for both test and train. If not, train-test sampling was done incorrectly.
  #I'll throw an error later
  dmy <- dummyVars(fmla, data = train) 
  train_dummy <- data.frame(predict(dmy, newdata = train))
  test_dummy <- data.frame(predict(dmy, newdata = test))
  train_target = train[,as.character(fmla[[2]])]
  test_target = test[,as.character(fmla[[2]])]
  
    # train the GBM model
  modelGbm <- train(fmla, data=train, method = 'gbm', trControl=control, verbose=FALSE)
  predictions.Gbm <- predict(modelGbm, newdata=test)
  pr.Gbm<-postResample(predictions.Gbm, test_target)
  pb$tick()
  
  # train the SVM model
  modelSvm <- train(x=train_dummy, y=train_target, method="svmRadial", trControl=control)
  predictions.SVM <- predict(modelSvm, newdata=test_dummy)
  pr.SVM <- postResample(predictions.SVM, test_target)
  pb$tick()
  
  # train the RF model
  modelRf <- train(fmla, data=train, method="rf", trControl=control)
  predictions.RF <- predict(modelRf, newdata=test)
  pr.RF <- postResample(predictions.RF, test_target)
  pb$tick()
  
  # train the RRF model
  modelRrf <- train(fmla, data=train, method="RRF", trControl=control)
  predictions.RRF <- predict(modelRrf, newdata=test)
  pr.RRF <- postResample(predictions.RRF, test_target)
  pb$tick()
  
  # train the xgLinear model
  # nthread=1 --> allow caret to handle this
  
  xgbGrid <- expand.grid(
    nrounds= c(250, 500, 1000),
    eta = c(0.01, 0.1, 0.2),
    lambda = c(0.01, 0.1, 1),
    alpha = c(.01, .1, 1)
  )
  
  modelXgl <- train(x=train_dummy, y=train_target, method="xgbLinear", trControl=control,
                    tuneGrid = xgbGrid,nthread=1)
  predictions.XGL <- predict(modelXgl, newdata=test_dummy)
  pr.XGL <- postResample(predictions.XGL, test_target)
  pb$tick()
  
  # train the xgTree model
  # nthread=1 --> allow caret to handle this
  
  xgbGrid <- expand.grid(
    nrounds = c(250, 500, 1000),
    max_depth = c(1, 2, 4, 6),
    eta = c(0.01, 0.1, 0.2),
    gamma = 0,
    colsample_bytree = c(1, 0.5, 0.25),
    min_child_weight = c(1, 2),
    subsample = 0.4
  ) 
  modelXgt <- train(x=train_dummy, y=train_target, method="xgbTree", trControl=control,
                    tuneGrid = xgbGrid, nthread=1)
  predictions.XGT <- predict(modelXgt, newdata=test_dummy)
  pr.XGT <- postResample(predictions.XGT, test_target)
  pb$tick()
  
  # collect resamples
  results <- resamples(list(GBM=modelGbm,
                            SVM=modelSvm,
                            RF=modelRf,
                            RRF=modelRrf,
                            XGL=modelXgl,
                            XGT=modelXgt))
  testResults <- t(data.frame(pr.Gbm,pr.SVM,pr.RF,pr.RRF,pr.XGL,pr.XGT))
  if(verbose){
    # summarize the distributions
    cat("RCV results\n\n")
    print(summary(results))
    cat("Test results\n\n")
    print(testResults)
  }
  if(plot){
    # boxplots of results
    print(bwplot(results))
    # dot plots of results
    print(dotplot(results) )
  }
  ret = list(finalTestResults=testResults,
             dummyTransform=dmy,
             gbm=list(model=modelGbm,bestTune=modelGbm$bestTune,testMetrics=pr.Gbm),
             svm=list(model=modelSvm,bestTune=modelSvm$bestTune,testMetrics=pr.SVM),
             rf=list(model=modelRf,bestTune=modelRf$bestTune,testMetrics=pr.RF),
             rrf=list(model=modelRrf,bestTune=modelRrf$bestTune,testMetrics=pr.RRF),
             xgl=list(model=modelXgl,bestTune=modelXgl$bestTune,testMetrics=pr.XGL),
             xgt=list(model=modelXgt,bestTune=modelXgt$bestTune,testMetrics=pr.XGT)
             )
  registerDoSEQ()
  return(ret)
}
