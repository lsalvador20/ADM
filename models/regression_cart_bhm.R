
# Install Packages 

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

# and Load the Libraries 

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
library(caTools)

# Set Up Project: Working Directory and Files   
getwd()


# Load data   
# read data into memory
data = read.csv("boston.csv")

# view data                                   
View(data)
head(data)

#
# EDA: Summary Statistics Analysis 
#
str(data)
summary(data)

# Check for missing values(1) and look how many unique values(2) 

sapply(data,function(x) sum(is.na(x)))  # (1)
sapply(data, function(x) length(unique(x))) # (2)

# Plot graph for missing values
missmap(data, main = "Missing values vs observed") # Use "Amelia" package

# Scatterplot for all data
plot(data)

#
# Preliminary Model Selection 1: Correlation matrix  #
# Explore data relationships
mcor <- cor(data)
round(mcor, digits=2)

# Get more visual Information
# Plot a correlation graph between dependent variable (medv) and all the rest of variables
# use "corrplot" package
cor1 = cor(data)
corrplot(cor1, method = "number")

# Fine-tune the models parameters. Set seed to get reproducible random results
# use seed before random data splitting
set.seed(123)

# For a stratified random sampling you can use caTools
split = sample.split(data$medv, SplitRatio = 0.7)
train = subset(data, split==TRUE)
test = subset(data, split==FALSE)



# Preliminary Model selection checks                        
# Lets double check that the dependent variable is normally distributed.  

x=train$medv #First lets assign dependent variable to x

# Check if the dependent variable is normally distributed
hist(x, freq = FALSE, col="grey", main="Histogram of Dependent Variable", xlab="medv")
xbar <-mean(x)
S=sd(x)
curve(dnorm(x,xbar,S), col="blue", add=T)

# Recording and Transforming variables          

train$logmedv <- log(train$medv)
test$logmedv <- log(test$medv)


# Lets check again our distribution
x = train$logmedv
hist(x, freq = FALSE, col="blue", main="Transformed Dependent Variable", xlab="logmedv")

# We can add a normal distribution line to the curve
xbar <-mean(x)
S=sd(x)
curve(dnorm(x,xbar,S), col="", add=T)

##################################
# Building the prediction model
##################################
RegressionModel <- lm(logmedv~crim+rm+dis+ptratio+lstat, data=train)
summary(RegressionModel)


# Lets check the variance inflation factor (VIF) to determine wheter multicollinearity
# vif > 5, there is collinearity associated with that variable: use package "car"
vif(RegressionModel)
plot(RegressionModel)

##########################################################
# Testing the prediction model - apply model to test data
###########################################################
prediction <- predict(RegressionModel, newdata = test)

head(prediction)

head(test$logmedv)


#####################################################
#    Model Performance                              #
#####################################################
# Evaluate the performance of the performance using nmetric function: use "rminer" package
mmetric(test$logmedv,prediction,c("MAE","RMSE","MAPE","RMSPE","RAE","RRSE","COR","R2"))



#############################################
#    Evaluate the accuracy of Model 
#############################################
# Regression diagnosis - R2 for the prediction model
# Multiple R-squared = 1 - SSE/SST

SSE <- sum((test$logmedv - prediction) ^ 2)
SSE
SST <- sum((test$logmedv - mean(test$logmedv)) ^ 2)
SST
1 - SSE/SST



##################################################
# DECISION TREE 
# Create a CART modelling - Regression
##################################################

# use "rpart" package 
tree = rpart(logmedv~crim+rm+dis+ptratio+lstat, data=train)
tree
rpart.plot(tree) # plot tree


printcp(tree) # display the results 
plotcp(tree) # visualize cross-validation results 


##########################################################
# Generate the prediction model - apply model to test data
###########################################################
treePredic <- predict(tree, newdata = test)

# Check values of the prediction, and compare it to the values of medv in the test data
head(treePredic)

head(test$logmedv)


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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Model Performance                              #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Evaluate the performance of the performance using nmetric function
mmetric(test$logmedv,treePredic,c("MAE","RMSE","MAPE","RMSPE","RAE","RRSE","COR","R2"))


# Cross-Validation
# Load libraries for cross-validation 
library(caret)
library(e1071)

# Number of folds
tr.control = trainControl(method = "cv", number = 10)

cp.grid = expand.grid( .cp = (0:10)*0.001)

1*0.001 

# Cross-validation
tr = train(logmedv~crim+rm+dis+ptratio+lstat, data=train, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)
tr


# Extract tree
best.tree = tr$finalModel
prp(best.tree)

printcp(best.tree) # display the results 

##########################################################
# Generate the prediction model - apply model to test data
###########################################################
# Make predictions
best.tree.pred = predict(best.tree, newdata=test)

# Check values of the prediction, and compare it to the values of medv in the test data
head(best.tree.pred)

head(test$logmedv)

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

# Evaluate the performance of the performance using nmetric function
mmetric(test$logmedv,best.tree.pred,c("MAE","RMSE","MAPE","RMSPE","RAE","RRSE","COR","R2"))









