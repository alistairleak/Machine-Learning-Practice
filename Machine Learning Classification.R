# Classification with machine learning

library(tidyverse)
library(caret)
library(MASS)
library(ranger)
library(rpart)
library(rpart.plot)
library(xgboost)

# Data Preparation
ml_data <- iris
ml_data <- ml_data %>% sample_frac(1)

split <- round(nrow(ml_data)*0.8)

ml_data_train <- ml_data[1:split,]
ml_data_test <- ml_data[(split+1):nrow(ml_data),]

# Classification based on Decison Tree

model_pt <- rpart(formula = Species~., 
                  data = ml_data_train)

pred_pt <- predict(model_pt, newdata = ml_data_test, type = "class")

rpart.plot(model_pt)
confusionMatrix(ml_data_test$Species, pred_pt)

# Classification based on Random Forest

model_rf <- ranger(formula = Species~., 
                   data = ml_data_train)

pred_rf <- predict(model_rf, data = ml_data_test)

confusionMatrix(ml_data_test$Species, pred_rf$predictions)


# Classification based on Gradient Boosting

lb <- as.numeric(ml_data_train$Species) - 1
num_class <- 3

modexgb <- xgboost(xgb_params, data = as.matrix(ml_data_train[-5]),
                   label = lb,
                   nrounds = 10, objective = "multi:softmax", num_class = num_class)

pred <- predict(modexgb, as.matrix(ml_data_test[, -5]))

confusionMatrix(as.numeric(ml_data_test$Species) - 1, pred)




