rm(list = ls(all = T))

setwd("C:\\Users\\NY 5211\\Downloads\\Analytics_Vidhya")

train_data = readRDS("trainData.rds")
test_data = readRDS("testData.rds")

test_data$RESPONDERS = NA
data = rbind(train_data, test_data)
colSums(is.na(data))
sum(is.na(data))

data = data[, !(colSums(is.na(data)) == nrow(data))]
data = data[,!(colSums(is.na(data[,-326])) > nrow(data[,-326]) * 0.7)]
data$CUSTOMER_ID = NULL
data$ZIP_CODE_FINAL = NULL
data$DC_ACTIVE_MON_01 = NULL
data$DC_ACTIVE_MON_02 = NULL
data$DC_ACTIVE_MON_03 = NULL
data$PA_PQ_TAG = NULL
data$NEFT_CC_CATE = NULL
data$NEFT_DC_CATEGORY = NULL
data$TPT_DC_CATEGORY_MON_01 = NULL
data$TPT_CC_CATEGORY_MON_01 = NULL
data$IMPS_CC_CATEGORY_MON_01 = NULL


library(DMwR)
imputed_data = centralImputation(data[,-85])
imputed_data$RESPONDERS = data$RESPONDERS

train = imputed_data[1:300000,]
test = imputed_data[300001:500000,]
test$RESPONDERS = NULL

train$RESPONDERS = ifelse(train$RESPONDERS == "Y", 1, 0)
train$RESPONDERS = as.factor(train$RESPONDERS)

set.seed(5211)

rows = createDataPartition(train$RESPONDERS,p = 0.7,list = F)
train1 = train[rows,]
test1 = train[-rows,]


### Oversampling
library(ROSE)
balance_train = ovun.sample(RESPONDERS ~ ., data = train1, N = 20000)
balance_train <- balance_train$data

library(randomForest)
library(caret)
random_model = randomForest(balance_train$RESPONDERS~., data = balance_train,keep.forest=TRUE, ntree=200)
random_model$importance

pred_train = predict(random_model, newdata = balance_train, type = "response", norm.votes = T)
confusionMatrix(pred_train, balance_train$RESPONDERS)

pred_validation = predict(random_model, newdata = test1, type = "response",norm.votes = T)
confusionMatrix(pred_validation, test1$RESPONDERS, positive = "1")

pred_actual = predict(random_model, newdata = test, type = "response", norm.votes = T)
write.csv(pred_actual, "Sample_submission_27.csv", row.names = F)
