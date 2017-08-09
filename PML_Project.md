# Practical Machine Learning Course Project

## Summary
Perform an analysis using the provided training dataset to train a model and 
predict the manner in which barbell lift excercises were performed in the
provided test dataset.

## Analysis
### Get Data

```r
library(caret)
library(rattle)

trainurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainurl, destfile = "train.csv")
download.file(testurl, destfile = "test.csv")

train <- read.csv("train.csv", na.strings = c("", "NA"))
test <- read.csv("test.csv", na.strings = c("", "NA"))
```

### Clean Data
Check that columns in training and test data sets match except for the final
column

```r
names(train) == names(test)
```

```
##   [1]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [12]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [23]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [34]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [45]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [56]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [67]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [78]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [89]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [100]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [111]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [122]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [133]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [144]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [155]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE
```

Check for NA values in the data

```r
nacnt <- sapply(train, function(y) sum(is.na(y)))
nacnt <- data.frame(nacnt)
```

Since the variables have either no NA values or are a majority NA values we will
remove the variables that are mostly NA and focus on the other variables.  We 
also remove the first few ID and timestamp variables that do not relate to the 
outcome we are trying to predict.

```r
i = 1
remove = c()

for (i in seq_len(nrow(nacnt))) {
      if (nacnt[i,1] > 0) { 
            remove = c(remove, i) 
      }
}

train <- train[,-remove]
train <- train[,-c(1:7)]

test <- test[,-remove]
test <- test[,-c(1:7)]
```

### Train Model
Partition the training data set into a training and validation set

```r
inTrain <- createDataPartition(y = train$classe, p = 0.70, list = FALSE)

dptrain <- train[inTrain,]
dpvalid <- train[-inTrain,]
```

Create several model fits using four different predicition methods

1. Decision Tree
2. Random Forest
3. Boosting with Trees
4. Linear Discriminate Analysis

We use the trControl parameter in the train function to include cross validation 
in each of the model fits and then predict on our validation dataset to get the 
out of sample error for each of the predition methods.

```r
set.seed(3141)

modfit1 <- train(classe ~ ., data = dptrain, method = "rpart", 
                 trControl = trainControl(method = "cv", number = 2))
modfit2 <- train(classe ~ ., data = dptrain, method = "rf", 
                 trControl = trainControl(method = "cv", number = 2))
modfit3 <- train(classe ~ ., data = dptrain, method = "gbm", 
                 trControl = trainControl(method = "cv", number = 2))
modfit4 <- train(classe ~ ., data = dptrain, method = "lda", 
                 trControl = trainControl(method = "cv", number = 2))

pred1 <- predict(modfit1, newdata = dpvalid)
pred2 <- predict(modfit2, newdata = dpvalid)
pred3 <- predict(modfit3, newdata = dpvalid)
pred4 <- predict(modfit4, newdata = dpvalid)

cm1 <- confusionMatrix(pred1, dpvalid$classe)
cm2 <- confusionMatrix(pred2, dpvalid$classe)
cm3 <- confusionMatrix(pred3, dpvalid$classe)
cm4 <- confusionMatrix(pred4, dpvalid$classe)
```

From the confusionMatrix we can calculate the out-of-sample error for each model 
and we see that the random forest method gives the best result, so we will use
that method to train our final model fit.

```r
1-cm1$overall['Accuracy'][[1]]
```

```
## [1] 0.5157179
```

```r
1-cm2$overall['Accuracy'][[1]]
```

```
## [1] 0.008326253
```

```r
1-cm3$overall['Accuracy'][[1]]
```

```
## [1] 0.03381478
```

```r
1-cm4$overall['Accuracy'][[1]]
```

```
## [1] 0.2944775
```

### Final Model and Test Prediction
Train the final model fit on the full training data set using the 
random forest method

```r
modfitfinal <- train(classe ~ ., data = train, method = "rf", 
                     trControl = trainControl(method = "cv", number = 2))
```

Predict the classe variable for the test dataset and print the results

```r
predfinal <- predict(modfitfinal, newdata = test)
print(predfinal)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusions

1. Of the 4 predictions methods tested the random forest method produced the
lowest error providing the best results and correctly predicted the classe
variable for the test dataset
2. Random forest may not be the best choice in terms of scalibility as it took a
greater amount of computation time to train the model.  Thus, it would be good 
to do further analysis of other model types if there were a desire to do this 
analysis on larger datasets.
