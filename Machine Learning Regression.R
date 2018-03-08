# Regression with machine learning

library(tidyverse)
library(caret)
library(MASS)
library(ranger)
library(rpart)
library(rpart.plot)
library(xgboost)


ml_data <- MASS::Boston
ml_data <- ml_data %>% sample_frac(1)

split <- round(nrow(ml_data)*0.8)

ml_data_train <- ml_data[1:split,]
ml_data_test <- ml_data[(split+1):nrow(ml_data),]

# Regression based on Decison Tree

model_pt <- rpart(formula = crim~., 
                   data = ml_data_train, method = "poisson")

pred_pt <- predict(model_pt, newdata = ml_data_test)

rpart.plot(model_pt)

plot(log(ml_data_test$crim+1), log(pred_pt+1))
cor(log(ml_data_test$crim+1), log(pred_pt+1))


# Classification based on Random Forest

model_rf <- ranger(formula = crim~., 
                  data = ml_data_train)

pred_rf <- predict(model_rf, data = ml_data_test)

plot(log(ml_data_test$crim), log(pred_rf$predictions))

# Classification based on XGBoost


modexgb <- xgboost(data = as.matrix(ml_data_train[-1]),
                   label = ml_data_train$crim,
                   nrounds = 20)

pred_xgb <- predict(modexgb, newdata = as.matrix(ml_data_test[-1]))

plot(log(ml_data_test$crim), log(pred_xgb))
cor(log(ml_data_test$crim), log(pred_xgb))

