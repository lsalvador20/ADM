#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Install Packages and Load the Library         #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

install.packages("Amelia")
install.packages("MASS")
install.packages("tree")
install.packages("DAAG")
install.packages("usdm")
install.packages("car")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("caret")
install.packages("lattice")
install.packages("ggplot2")
install.packages("e1071")
install.packages("bootstrap")
install.packages("corrplot")
install.packages("ggplot2")
install.packages("reshape2")
install.packages("plyr")
install.packages("caTools")
install.packages("rminer")

library(plyr)
library(caTools) 
library(reshape2)
library(corrplot)
library(usdm)
library(car)
library(Amelia)
library(MASS)
library(DAAG)
library(gmodels)
library(rpart)
library(rpart.plot)
library(lattice)
library(ggplot2)
library(caret)
library(tree)
library(rminer)





detach("package:usdm", unload=TRUE)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Set Up Project: Working Directory and Files   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
getwd()
# Set working Dir
setwd("E:/DATA/Academics/R")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Load DATA                                     #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
data = read.csv("Test/boston.csv")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    View DATA                                     #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
View(data)
head(data)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    EDA: Summary Statistics Analysis              #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

str(data)
summary(data)


# Check for missing values(1) and look how many unique values(2) 

sapply(data,function(x) sum(is.na(x)))  # (1)
sapply(data, function(x) length(unique(x))) # (2)

# Plot graph
missmap(data, main = "Missing values vs observed")


# Scatterplot for all data
plot(data)




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Preliminary Model Selection 1: Correlation matrix  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
mcor <- cor(data)
round(mcor, digits=2)

# Plot a correlation graph between I.V
newdatacor = cor(data[1:13])
corrplot(newdatacor, method = "number")

# Plot a correlation graph between D.P (medv) and all the rest I.V
newdatacor1 = cor(data)
corrplot(newdatacor1, method = "number")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Building the prediction model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# seed(1234) is important, allows for us to recuperate the results exactly as R 
# produced them. Use seeds before random data splitting

set.seed(123)

#split the dataset into a training (for building the model) and testing (for testing the model performance)
# For a stratified random sampling you can use caTools
split = sample.split(data$medv, SplitRatio = 0.7)
train = subset(data, split==TRUE)
test = subset(data, split==FALSE)

##########################################
# Create linear regression model
##########################################
predModel <-lm(medv~indus + rm + tax + ptratio + lstat, data=train)


# Summary of the model
summary(predModel)

plot(predModel)
par(mfrow=c(2,2))
plot(predModel)

##########################################################
# Testing the prediction model - apply model to test data
###########################################################

prediction1 <- predict(predModel, newdata = test)


# Check values of the prediction, and compare it to the values of medv in the test data
head(prediction1)
#     5        8       15       16       18       19 
#     31.32580 21.43988 21.02959 20.59517 18.06557 16.81720

head(test$medv)
# [1] 36.2 27.1 18.2 19.9 17.5 20.2


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Preliminary Model selection checks                        #
# Lets double check that the dependent variable is normally distributed.  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# First lets assign dependent variable to x
x=train$medv


# Check if the dependent variable is normally distributed
hist(x, freq = FALSE, col="grey", main="Histogram of Dependent Variable", xlab="medv")
xbar <-mean(x)
S=sd(x)
curve(dnorm(x,xbar,S), col="blue", add=T)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Recording and Transforming variables          #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

train$logmedv <- log(train$medv)

test$logmedv <- log(test$medv)





# Lets check again our distribution
x = train$logmedv
hist(x, freq = FALSE, col="blue", main="Transformed Dependent Variable", xlab="logmedv")

# We can add a normal distribution line to the curve
xbar <-mean(x)
S=sd(x)
curve(dnorm(x,xbar,S), col="", add=T)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Preliminary Model Selection 2: Stepwise selection  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

step(lm(logmedv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data=train), direction="both")


predModelSt <- lm(logmedv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat, data=train)
summary(predModelSt)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Model Performance                              #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Lets check the variance inflation factor (VIF) to determine wheter multicollinearity
# vif > 5, there is collinearity associated with that variable

vif(predModelSt)


# Residual plot how much teh variance of the estimated coefficients are increased over the case of no correlation among the X variables
plot(predModelSt)
par(mfrow=c(2,2))
plot(predModelSt)


##########################################################
# Testing the prediction model - apply model to test data
###########################################################

prediction2 <- predict(predModelSt, newdata = test)

#head(test)
# Check values of the prediction, and compare it to the values of medv in the test data
head(prediction2)
# 5        8       15       16       18       19 
# 3.352656 2.898699 2.995071 3.013024 2.861309 2.903106 

head(test$logmedv)
# [1] 3.589059 3.299534 2.901422 2.990720 2.862201 3.005683


#############################################
#    Evaluate the accuracy of Model 
#############################################

# Regression diagnosis - R2 for the prediction model
# Multiple R-squared = 1 - SSE/SST 

SSE <- sum((test$logmedv - prediction2) ^ 2)
SSE
SST <- sum((test$logmedv - mean(test$logmedv)) ^ 2)
SST
1 - SSE/SST


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Model Performance                              #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Evaluate the performance of the performance using nmetric function
mmetric(test$logmedv,prediction2,c("MAE","RMSE","MAPE","RMSPE","RAE","RRSE","COR","R2"))
# MAE       RMSE       MAPE      RMSPE        RAE       RRSE        COR         R2 
# 0.1442153  0.2121304  5.0483518  0.7556934 51.0707953 54.5545551  0.8432837  0.7111274 


############################
predictionModelFinal <- lm(logmedv~crim+rm+dis+ptratio+lstat, data=train)
summary(predictionModelFinal)

vif(predictionModelFinal)
plot(predictionModelFinal)

predictionMF <- predict(predictionModelFinal, newdata = test)

head(predictionMF)
#5        8       15       16       18       19 
#3.376955 2.903045 3.024919 3.049910 2.866227 2.904409 

head(test$logmedv)
#[1] 3.589059 3.299534 2.901422 2.990720 2.862201 3.005683

## Evaluate the performance of the performance using nmetric function
mmetric(test$logmedv,predictionMF,c("MAE","RMSE","MAPE","RMSPE","RAE","RRSE","COR","R2"))
# MAE       RMSE       MAPE      RMSPE        RAE       RRSE        COR         R2 
# 0.1572324  0.2291411  5.5122488  0.8154484 55.6805123 58.9292699  0.8154168  0.6649045 


SSE <- sum((test$logmedv - predictionMF) ^ 2)
SSE
SST <- sum((test$logmedv - mean(test$logmedv)) ^ 2)
SST
1 - SSE/SST



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# K-fold cross-validation
#library(DAAG)
cv.lm(data = train, modelMR, m=3) # 3 fold cross-validation

# Assessing R2 shrinkage using 10-Fold Cross-Validation 

model1 <-lm(logmedv~rm+lstat+crim+zn+chas+dis, data = trainData)

library(bootstrap)
# define functions 
theta.model1 <- function(x,y){lsfit(x,y)}
theta.predict <- function(fit,x){cbind(1,x)%*%model1$coef} 

# matrix of predictors
X <- as.matrix(trainData[c("rm","lstat","crim", "zn", "chas", "dis")])
# vector of predicted values
y <- as.matrix(trainData[c("logmedv")]) 

results <- crossval(X,y,theta.model1,theta.predict,ngroup=10)
cor(y, model1$fitted.values)**2 # raw R2 
cor(y,results$cv.model1)**2 # cross-validated R2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DECISION TREE 
# Create a CART modelling - Regression
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tree = rpart(logmedv~crim+rm+dis+ptratio+lstat, data=train)
tree
rpart.plot(tree)
#prp(tree)


# Added
printcp(tree) # display the results 
plotcp(tree) # visualize cross-validation results 


##########################################################
# Generate the prediction model - apply model to test data
###########################################################
treePredic <- predict(tree, newdata = test)

# Check values of the prediction, and compare it to the values of medv in the test data
head(treePredic)
# 5        8       15       16       18       19 
# 3.460960 2.880228 3.008949 3.171211 3.008949 3.008949
head(test$logmedv)
# [1] 3.589059 3.299534 2.901422 2.990720 2.862201 3.005683

#############################################
#    Evaluate the accuracy of Model 
#############################################

# Regression diagnosis - R2 for the prediction model
# Multiple R-squared = 1 - SSE/SST 

SSE <- sum((test$logmedv - treePredic) ^ 2)
SSE
SST <- sum((test$logmedv - mean(test$logmedv)) ^ 2)
SST
1 - SSE/SST

MSE <- mean((treePredic - test$logmedv)^2)
MSE
# [1] 0.0561333

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Model Performance                              #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Evaluate the performance of the performance using nmetric function
mmetric(test$logmedv,treePredic,c("MAE","RMSE","MAPE","RMSPE","RAE","RRSE","COR","R2"))
# MAE       RMSE       MAPE      RMSPE        RAE       RRSE        COR         R2 
# 0.1749846  0.2369247  6.1370146  0.8697003 61.9670747 60.9310063  0.7958823  0.6334286 


# Added
printcp(tree) # display the results 
plotcp(tree) # visualize cross-validation results 
summary(tree) # detailed summary of splits


# # Load libraries for cross-validation ### We have second way below
library(caret)
library(e1071)

# Number of folds
tr.control = trainControl(method = "cv", number = 10)

# cp values
cp.grid = expand.grid( .cp = (0:10)*0.001)

# What did we just do?
1*0.001 


# Cross-validation
tr = train(logmedv~crim+rm+dis+ptratio+lstat, data=train, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)
tr


# Extract tree
best.tree = tr$finalModel
prp(best.tree)

printcp(best.tree) # display the results 
#plotcp(best.tree) # visualize cross-validation results 




##########################################################
# Generate the prediction model - apply model to test data
###########################################################
# Make predictions
best.tree.pred = predict(best.tree, newdata=test)

# Check values of the prediction, and compare it to the values of medv in the test data
head(best.tree.pred)
# 5        8       15       16       18       19 
# 3.460960 2.950724 3.008949 3.171211 3.008949 3.008949
head(test$logmedv)
# [1] 3.589059 3.299534 2.901422 2.990720 2.862201 3.005683



#############################################
#    Evaluate the accuracy of Model 
#############################################

# Regression diagnosis - R2 for the prediction model
# Multiple R-squared = 1 - SSE/SST 

SSE_tree = sum((best.tree.pred - test$logmedv)^2)
SSE_tree
SST_tree <- sum((test$logmedv - mean(test$logmedv)) ^ 2)
SST
1 - SSE_tree/SST_tree

mean((best.tree.pred - test$logmedv)^2)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Model Performance                              #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Evaluate the performance of the performance using nmetric function
mmetric(test$logmedv,best.tree.pred,c("MAE","RMSE","MAPE","RMSPE","RAE","RRSE","COR","R2"))
# MAE       RMSE       MAPE      RMSPE        RAE       RRSE        COR         R2 
# 0.1605663  0.2279841  5.6235237  0.8381510 56.8611399 58.6317224  0.8131218  0.6611670 


########## DS) 500

tree_model <- tree(logmedv~crim+rm+dis+ptratio+lstat, data=train)
tree_model
plot(tree_model)
text(tree_model)

tree_predict <- predict(tree_model, newdata=test)
MSE <- mean((tree_predict - test$logmedv)^2)
MSE
# [1] 0.0561333

# Pruning by Cross-validation 
# Or CV
cv_tree = cv.tree(tree_model)

names(cv_tree)
plot(cv_tree$size,
     cv_tree$dev,
     type = "b",
     xlab = "Tree size",
     ylab ="MSE")

which.min(cv_tree$size)
cv_tree$size[1]
# [1] 8

# pruned model 
pruned_tree <- prune.tree(tree_model, best=5)
plot(pruned_tree)
text(pruned_tree)


# check accuracy of the model 
tree_predic <- predict(pruned_tree, newdata=test)
MSE <- mean((tree_predic - test$logmedv)^2)
MSE
# [1] 0.0666494


# https://www.r-bloggers.com/using-neural-networks-for-credit-scoring-a-simple-example/
# https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Neural Network
# Create a ANN modelling - Regression
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
install.packages("neuralnet")
library("neuralnet")

## build the neural network (NN)
creditnet <- neuralnet(logmedv~crim+rm+dis+ptratio+lstat, data=train, hidden = 4, lifesign = "minimal", 
                       linear.output = FALSE, threshold = 0.1)


net_regression <- neuralnet(train$logmedv ~ LTI + age, trainset, hidden = 4, lifesign = "minimal", 
                       linear.output = FALSE, threshold = 0.1)

