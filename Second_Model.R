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
set.seed(5211)
rows = createDataPartition(train_data$RESPONDERS , p = 0.7, list = F)
train = train_data[rows,]
test = train_data[-rows,]


### Oversampling
library(ROSE)
balance_train = ovun.sample(RESPONDERS ~ ., data = train,method = "both", N = 100000)
balance_train <- balance_train$data

library(rpart)

cart_model = rpart(balance_train$RESPONDERS~., balance_train ,method = "class")
printcp(cart_model)
plot(cart_model)

cart_model$variable.importance

### Standardization

preProc = preProcess(balance_train[, setdiff(names(balance_train),"RESPONDERS")])
balance_train = predict(preProc, balance_train)
test = predict(preProc, test)

library(doParallel)
registerDoParallel(8)

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

