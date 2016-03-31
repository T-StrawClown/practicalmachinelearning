ds_raw.tr <- read.csv("pml-training.csv")
ds_raw.ts <- read.csv("pml-testing.csv")

# use only attributes of testing set for training
pred_cols <- names(ds_raw.ts[, colSums(is.na(ds_raw.ts)) < nrow(ds_raw.ts)])
pred_cols <- pred_cols[pred_cols != "problem_id"]
ds.tr <- ds_raw.tr[,c("classe", pred_cols)]

#remove non-numeric columns
require(plyr)
require(dplyr)
ds.tr <- ds.tr %>%
        select(-cvtd_timestamp,
               -X,
               -user_name,
               -raw_timestamp_part_1,
               -raw_timestamp_part_2,
               -new_window,
               -num_window)

set.seed(42);
#create validation set and training set
require(caret)
inTrain <- createDataPartition(y = ds.tr$classe, p = .6, list = FALSE)
ds.tr.tr <- ds.tr[inTrain,]
ds.tr.vl <- ds.tr[-inTrain,]

# do PCA 
ds.tr.pca <- preProcess(x = ds.tr.tr[,-1], method = c("pca"), thresh = .95)
ds.training <- predict(ds.tr.pca, ds.tr.tr)
ds.validation <- predict(ds.tr.pca, ds.tr.vl)

tr_control = trainControl(method = "cv",
                          number = 6)

# random forest, 6 fold cross-validation
require(randomForest)
grid.rf <- expand.grid(mtry = c(2^(1:4), 25))
mdl.rf <- train(classe ~ ., data = ds.training, method = "rf", trControl = tr_control, tuneGrid = grid.rf)
# mdl.rf <- randomForest(classe ~ ., data = ds.training, mtry = 2, ntree = 200)
pred.rf <- predict(mdl.rf$finalModel, newdata = ds.validation)
conf.rf <- confusionMatrix(pred.rf, ds.validation$classe)

# SVM, 6 fold cross-validation
require(e1071)
# mdl.svm.tuned <- tune.svm(classe ~ ., data = ds.training, gamma = 10^(-3:-1), cost = 10^(-1:1))
mdl.svm <- svm(classe ~ ., data = ds.training, cross = 6, gamma = .1, cost = 10)
pred.svm <- predict(mdl.svm, ds.validation)
conf.svm <- confusionMatrix(pred.svm, ds.validation$classe)

# C5.0, 6 fold cross-validation
require(C50)
# mdl.c50 <- C5.0(classe ~ ., data = ds.training)
grid.c50 <- expand.grid(winnow = c(TRUE,FALSE), trials=c(5,10,20), model = "tree")
mdl.c50 <- train(classe ~ ., data = ds.training, method = "C5.0", trControl = tr_control, tuneGrid = expand.grid(winnow = TRUE, trials = 20, model = "tree"))
pred.c50 <- predict(mdl.c50$finalModel, newdata = ds.validation)
conf.c50 <- confusionMatrix(pred.c50, ds.validation$classe)

# combining predictors
inValTrain <- createDataPartition(y = ds.validation$classe, p = .5, list = FALSE)
ds.validation.tr <- ds.validation[inValTrain,]
ds.validation.ts <- ds.validation[-inValTrain,]

ds.combined <- data.frame(classe = ds.validation.tr$classe,
                          rf = predict(mdl.rf$finalModel, newdata = ds.validation.tr),
                          svm = predict(mdl.svm, newdata = ds.validation.tr),
                          c50 = predict(mdl.c50$finalModel, newdata = ds.validation.tr))

# final model
#mdl.final.rf <- train(classe ~ ., ds.validation.tr, method = "rf", trControl = tr_control, tuneGrid = grid.rf)
mdl.final.svm <- svm(classe ~ ., data = ds.validation.tr, cross = 6, gamma = .06, cost = 10)
#mdl.final.c50 <- too slow to run 
pred.final <- predict(mdl.final.svm, newdata = ds.validation.ts)
conf.final <- confusionMatrix(pred.final, ds.validation.ts$classe)

# run model on test data
ds.ts <- ds_raw.ts[,c("problem_id", pred_cols)]
ds.testing <- predict(ds.tr.pca, ds.ts[,-(2:8)])
pred.results <- predict(mdl.final.svm, newdata = ds.testing[,-1])
ds.results <- data.frame(problem_id = ds.testing$problem, result = as.factor(pred.results))


