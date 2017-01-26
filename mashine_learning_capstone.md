

made by Svetlana Tabulina

---
title: "Machine learning capstone analysis"
output: html_document
---



## Executive summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively.  Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 
exactly according to the specification (Class A), 
throwing the elbows to the front (Class B), 
lifting the dumbbell only halfway (Class C), 
lowering the dumbbell only halfway (Class D) 
throwing the hips to the front (Class E).

The goal of project is to predict the manner in which they did the exercise.

There are two datasets avaliable:
training set - 19622 observations of  160 variables, including "classe"  factor variable (fashions) to build the model

testing set - 20 observations to predict 20 different test cases

## Download and read data


```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile = "./training.csv")

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile = "./testing.csv")
```

As i see from the csv files downloaded, there are tree strings "NA","","#DIV/0!" that should be read as NA.
To avoid numeric variables readed as factors, i set stringAsFactors=FALSE, and set needed varibles class manually.


```r
captraining<-read.csv("./training.csv",sep=",",dec=".",stringsAsFactors = FALSE,na.strings = c("NA","","#DIV/0!"))
captraining$user_name<-as.factor(captraining$user_name)
captraining$cvtd_timestamp<-dmy_hm(captraining$cvtd_timestamp)
captraining$new_window<-as.factor(captraining$new_window)
captraining$classe<-as.factor(captraining$classe)

testing<-read.csv("./testing.csv",sep=",",dec=".",stringsAsFactors = FALSE,na.strings = c("NA","","#DIV/0!"))

testing$user_name<-as.factor(testing$user_name)
testing$cvtd_timestamp<-dmy_hm(testing$cvtd_timestamp)
testing$new_window<-as.factor(testing$new_window)
```

To deal with NA i delete columns with more then 50% NA. 60 variables remained and there are no more NA.


```r
training<-captraining[, -which(colMeans(is.na(captraining)) > 0.5)]
dim(training)
```

```
## [1] 19622    60
```

```r
sum(complete.cases(training))
```

```
## [1] 19622
```


## Exploratory analysis


```r
qplot(X,classe,data=training,color=classe,main="Class of observation versus variable X")
```

![](mashine_learning_capstone_files/figure-html/exploratory analysis-1.png)<!-- -->

```r
cc2<- training %>%
  select(cvtd_timestamp,classe, user_name) %>%
  group_by(cvtd_timestamp, classe, user_name) %>%
  summarize(count=n())

qplot(x=cvtd_timestamp, y=count, data=cc2, color=classe, shape=user_name, main="Number of observations of  each class \n by timestamp and user name")
```

![](mashine_learning_capstone_files/figure-html/exploratory analysis-2.png)<!-- -->



From this plots we can see, that rows are sorted by class, and X is row index.
THis means that X shouldnt be used in model building.

Timestamp and user name plot shows, that we dont need to care about time slices, it seems what each user make  repetitions in his own day and time.

So in my analysis i will use all variables, that dont have NA values, excluding X.

## Cross validation and model 

I have one set to build a model (training set), and will use k-folds cross validation with k=10 to train and test the model.


```r
train_control<- trainControl(method="cv", number=10, savePredictions = TRUE)
```

I decided to start from decisions tree. Set the seed (101):


```r
training<-training[,2:60]

set.seed(101)

model_rpart<- train(classe~., data=training, trControl=train_control, method="rpart")
predictions_rpart<- predict(model_rpart,training)
```



```r
cm<-confusionMatrix(predictions_rpart,training$classe)
cm$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.4955662      0.3406638      0.4885461      0.5025876      0.2843747 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

```r
cm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 5080 1581 1587 1449  524
##          B   81 1286  108  568  486
##          C  405  930 1727 1199  966
##          D    0    0    0    0    0
##          E   14    0    0    0 1631
```

Accuracy is not sufficient.
Next try is linear discriminant analysis: 


```r
set.seed(101)

model_lda<- train(classe~., data=training, trControl=train_control, method="lda")
predictions_lda<- predict(model_lda,training)
```



```r
cm<-confusionMatrix(predictions_lda,training$classe)
cm$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.486495e-01   6.816494e-01   7.425179e-01   7.547073e-01   2.843747e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00  2.214021e-183
```

```r
cm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 4776  531  299  190  138
##          B  153 2529  310  118  418
##          C  278  481 2377  383  228
##          D  371  128  386 2478  293
##          E    2  128   50   47 2530
```


Accuracy is better, but us I need at least 80% correct answers to pass the test, i need at least 80% accuracy.
Lets try boosting


```r
set.seed(101)

model_gbm<- train(classe~., data=training, trControl=train_control, method="gbm")
predictions_gbm<- predict(model_gbm,training)
```



```r
cm<-confusionMatrix(predictions_gbm,training$classe)
cm$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9982673      0.9978083      0.9975795      0.9987997      0.2843747 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

```r
cm$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    1    0    0    0
##          B    0 3791    3    0    0
##          C    0    5 3414   11    0
##          D    0    0    5 3200    4
##          E    0    0    0    5 3603
```

Accuracy is perfect!
In the confusion matrix above there is all information about this model.
Expected out of sample error for testing data should be about 1% (not dramatically bigger)

Lets predict the class for testing data:


```r
predict(model_gbm,testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The Course Project Prediction Quiz shows that my model predict 20 cases from 20!








