rm(list = ls(all = T))

setwd("C:\\Users\\NY 5211\\Downloads\\Analytics_Vidhya")
file_train = read.csv(file = "train.csv", header = T, na.strings = "")
saveRDS(object = file_train, file = "trainData.rds")

file_test = read.csv(file = "test_plBmD8c.csv", header = T, na.strings = "")
saveRDS(object = file_test, file = "testData.rds")

file_test$RESPONDERS = NA
data = rbind(file_train, file_test)
colSums(is.na(data))
sum(is.na(data))

data = data[, !(colSums(is.na(data)) == nrow(data))]
data = data[,!(colSums(is.na(data[,-326])) > nrow(data[,-326]) * 0.7)]
data$CUSTOMER_ID = NULL
data$ZIP_CODE_FINAL = NULL

library(DMwR)
imputed_data = centralImputation(data[,-93])
imputed_data$RESPONDERS = data$RESPONDERS

factor_type = imputed_data[, sapply(imputed_data[,-93], is.factor)]
factor_type$RESPONDERS = NULL
fac_col = factor_type

library(dummies)
factor_type = sapply(factor_type, dummy)
cust_data = imputed_data[, setdiff(names(imputed_data), names(fac_col))]

cust = data.frame(cust_data, factor_type)

train_data = cust[1:300000,]
test_data = cust[300001:500000,]
test_data$RESPONDERS = NULL

train_data$RESPONDERS = ifelse(train_data$RESPONDERS == "Y", 1, 0)
train_data$RESPONDERS = as.factor(train_data$RESPONDERS)

library(caret)
rows = createDataPartition(train_data$RESPONDERS , p = 0.7, list = F)
train = train_data[rows,]
test = train_data[-rows,]


### Oversampling
library(ROSE)
balance_train = ovun.sample(RESPONDERS ~ ., data = train, N = 10000)
balance_train <- balance_train$data

#### Logisitc Regression #######################################################################################
glm_model = glm(formula = balance_train$RESPONDERS~., family = binomial,data = balance_train)
summary(glm_model)

library(ggplot2)
library(ROCR)

pred_train = predict(glm_model, newdata = balance_train,type = "response")
prob = prediction(pred_train, balance_train$RESPONDERS)
tprfpr = performance(prob, "tpr","fpr")
plot(tprfpr, col = rainbow(10), colorize = T, print.cutoffs.at=seq(0,1,0.05))

pred_auc = performance(prob, measure = "auc")
auc = pred_auc@y.values[[1]]

## Predict on Train
pred_train_class = ifelse(pred_train > 0.5,1,0)
table(balance_train$RESPONDERS, pred_train_class)
confusionMatrix(pred_train_class, balance_train$RESPONDERS, positive = "1")

### Predict on Test

pred_test = predict(glm_model, test, type = "response")
pred_valid = ifelse(pred_test>0.5,1,0)
confusionMatrix(pred_valid,test$RESPONDERS,positive = "1")

## Predict on Actual 

pred_actual = predict(glm_model, newdata = test_data, type = "response")
submission = ifelse(pred_actual > 0.5, 1, 0)
write.csv(submission, "sample_submission_5.csv", row.names = F)

#### Regularization ###########################################################################
x = model.matrix(balance_train$RESPONDERS~., balance_train)
head(x)

library(glmnet)

fit.lasso = glmnet(x, balance_train$RESPONDERS, family = "binomial", alpha = 1)

fit.lasso.cv = cv.glmnet(x, balance_train$RESPONDERS, type.measure = "mse", alpha = 1,
                         family = "binomial", nfolds = 5, parallel = T)

par(mfrow = c(3,2))
plot(fit.lasso)
plot(fit.lasso.cv)

newmodel_lasso = glmnet(x, balance_train$RESPONDERS, family = "binomial", type.multinomial = "grouped",lambda = fit.lasso.cv$lambda.min)
head(newmodel_lasso$classnames)

## Predict on Test Data
x.test = model.matrix(test$RESPONDERS~., test)
y.test = test$RESPONDERS

pred_lasso = predict(fit.lasso, x.test, family = "binomial", type = "class", levels = T)
confusionMatrix(pred_lasso,y.test, positive = "1")

## Predict on Actual Test
test_data$RESPONDERS = 0
a = model.matrix(test_data$RESPONDERS~., test_data)




pred_lasso_validate = predict(newmodel_lasso,a, test_data$RESPONDERS, s = fit.lasso.cv$lambda.min, family = "binomial",
                              type = "class", levels = T)
write.csv(pred_lasso_validate, "sample_submission_2.csv", row.names = F)

### Ridge
fit.ridge = glmnet(x, train$RESPONDERS, family = "binomial", alpha = 0)

##fit.ridge.cv = cv.glmnet(x,train$RESPONDERS, type.measure = "mse", alpha = 0,
#  family = "binomial", nfolds = 10, parallel = T)
plot(fit.ridge)
plot(fit.ridge.cv)

newmodel_ridge = glmnet(x, train$RESPONDERS, family = "binomial", type.multinomial = "grouped",lambda = fit.ridge.cv$lambda.min)
head(newmodel_ridge$classnames)

pred_ridge = predict(newmodel_ridge, x.test, s = newmodel_ridge$lambda.min, family = "binomial", type = "class", levels = T)
confusionMatrix(pred_ridge,y.test, positive = "1")



### PCA #######################################################################################
pca =prcomp(balance_train[,-54])
summary(pca)
### Manually Calculate CUmulative Propertion
vars = apply(pca$x, 2, var)
props = vars/sum(vars)
cumsum(props)

pca_train = data.frame(pca$x[,1:30], balance_train$RESPONDERS)
pca_model = glm(pca_train$balance_train.RESPONDERS~., family = binomial, data = pca_train)
summary(pca_model)

pca_test = predict(pca, newdata = test[,-54])
pca_test = data.frame(pca_test[,1:30], test$RESPONDERS)

pred_pca_train = predict(pca_model, newdata = pca_train, type = "response")
pred_pca_class = ifelse(pred_pca_train > 0.5, 1 , 0)
confusionMatrix(pred_pca_class, pca_train$balance_train.RESPONDERS)

pred_pca_test = predict(pca_model, newdata = pca_test, type = "response")
pred_pca_class2 = ifelse(pred_pca_test > 0.5, 1, 0)
confusionMatrix(pred_pca_class2, pca_test$test.RESPONDERS)
   
pca_test_actual = predict(pca, test_data)
pca_test_actual = data.frame(pca_test_actual[,1:30])

pca_actual = predict(pca_model, pca_test_actual, type = "response")
pca_class = ifelse(pca_actual > 0.5,1,0)
write.csv(pca_class, "sample_submission_7.csv", row.names = F)

### Decision Trees #######################################################################################
library(rpart)
cart_model = rpart(balance_train$RESPONDERS~., balance_train ,method = "class")
printcp(cart_model)
plot(cart_model)

cart_model$variable.importance

pred_cart_train = predict(cart_model, test, type = "class")
confusionMatrix(pred_cart_train, test$RESPONDERS, positive = "1")

pred_cart_actual = predict(cart_model, test_data, type = "class")
write.csv(pred_cart_actual, "sample_submission_5.csv", row.names = F)


### C5.0 #######################################################################################

library(C50)
c5_model = C5.0(train$RESPONDERS~., data = train)

##C5imp(c5_model, metric = "usage")

pred_c5_test = predict(c5_model, test)
confusionMatrix(pred_c5_test, test$RESPONDERS)

pred_c5_actual = predict(c5_model, test_data)
write.csv(pred_c5_actual, "sample_submission_2.csv", row.names = F)


### Randomm Forest #######################################################################################
library(randomForest)
model_random = randomForest(balance_train$RESPONDERS~., data = balance_train,  keep.forest=TRUE,ntree=100)
print(model_random)
model_random$importance

pred_random_train = predict(model_random, newdata = balance_train, type = "response", norm.votes = T)
confusionMatrix(pred_random_train, balance_train$RESPONDERS, positive = "1")

pred_random_test = predict(model_random, newdata = test, type = "response", norm.votes = T)
confusionMatrix(pred_random_test, test$RESPONDERS, positive = "1")

### Actual Data
random_actual_test = predict(model_random, newdata = test_data, type = "response", norm.votes = T)
write.csv(random_actual_test, "sample_submission_17.csv", row.names = F)


#### ADA BOOST #######################################################################################
library(ada)

model_adaBoost = ada(x = balance_train[,-54], y = balance_train$RESPONDERS,iter = 50, loss = "logistic")

model_adaBoost
summary(model_adaBoost)

pred_ada_train = predict(model_adaBoost, balance_train[,-54])
confusionMatrix(pred_ada_train, balance_train$RESPONDERS, positive = "1")

pred_ada_validation = predict(model_adaBoost, test[,-54])
confusionMatrix(pred_ada_validation, test$RESPONDERS, positive = "1")

pred_ada_actual = predict(model_adaBoost, test_data)
write.csv(pred_ada_actual,  "sample_submission_20.csv", row.names = F)

### SVM MODEL #######################################################################################
library(e1071)

model_svm = svm(balance_train$RESPONDERS~., data = balance_train, kernel= "radial")
summary(model_svm)

svm_train = predict(model_svm,  balance_train)
confusionMatrix(svm_train,  balance_train$RESPONDERS, positive = "1")

svm_validation = predict(model_svm, test)
confusionMatrix(svm_validation, test$RESPONDERS)

svm_actual_test = predict(model_svm, test_data)
write.csv(svm_actual_test, "sample_submission_21.csv", row.names = F)

### XGBOOST #######################################################################################
library(xgboost)
dtrain = xgb.DMatrix(data = as.matrix(balance_train[,-54]),
                     label = balance_train$RESPONDERS)

model_xgboost = xgboost(data = dtrain, max.depth = 2, 
                        eta = 1, nthread = 2, nround = 2,verbose = 1)

dtest = xgb.DMatrix(data = as.matrix(test[,-54]),
                    label = test$RESPONDERS)

watchlist = list(balance_train = dtrain, test = dtest)

model = xgb.train(data=dtrain, max.depth=2,
                  eta=1, nthread = 2, nround=5, 
                  watchlist=watchlist,
                  eval.metric = "error", 
                  verbose=1)

xg_train = predict(model, as.matrix(balance_train[,-54]))
xg_validation = predict(model, as.matrix(test[,-54]))

pred_xg_validate = ifelse(xg_validation > 0.5,1,0)
confusionMatrix(pred_xg_validate, test$RESPONDERS)

### Actual
xg_actual = predict(model, as.matrix(test_data))
pred_actual_xg = ifelse(xg_actual > 0.8, 1, 0)
write.csv(pred_actual_xg, "sample_submission_22..csv", row.names = F)


### GRID SEARCH ON XGBOOST
ctrl <- trainControl(method = "repeatedcv",   # n fold cross validation
                     number = 5,							# do 5 repititions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE,
                     allowParallel = TRUE)

xgb.grid <- expand.grid(nrounds = c(2,5), #the maximum number of iterations
                        eta = c(1), # shrinkage
                        max_depth = c(2,5,8),
                        subsample = 1,
                        gamma = 0,               #default=0
                        colsample_bytree = 1,    #default=1
                        min_child_weight = 1)     #default=1)

install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")

xgb.tune <-train(x=balance_train[,-54],
                 y=balance_train$RESPONDERS,
                 method="xgbTree",
                 metric="ROC",
                 trControl=ctrl,
                 tuneGrid=xgb.grid)
