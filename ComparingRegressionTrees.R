### ISLR Lab
### Regression Trees
install.packages('tree')
install.packages('ISLR2')

# This tree will use regression to model the median value of owner-occupied homes (medv) contained in the Boston dataset
library(tree)
library(ISLR2)
str(Boston)
?Boston

Model <- c('Single Tree', 'Bagged Trees', 'Random Forest')
MSE <- c(NA,NA,NA)
RMSE <- c(NA,NA,NA)
results <- data.frame(Model, MSE, RMSE)
results

## Create a training set, using half the rows
?tree

set.seed(42)
train <- sample(x = 1:nrow(Boston), size = nrow(Boston)/2)
tree.boston <- tree(medv ~ ., data = Boston, subset = train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston, pretty = 0)

## 4 variables have been used
## 6 splits
## 7 terminal nodes
## Residual mean deviance: 10.38
## Deviance is the number of misclassifications

## Splits are determined in a 'greedy' fashion, i.e. to minimize the residual sum of squares (RSS) at that juncture, without a considering the implications on future splits (i.e. interaction effects)
## Splits define the sample space (or subspace) for which future analyses / decision models will consider

## Increasing model complexity (i.e. the number of internal and terminal nodes) can improve fit to the training data,
## and overfitting, so that the model will underperform on new data

## Pruning a decision tree will minimize the RSS/SSE of the model while maximizing parsimony, and allowing for generalization
## Cost complexity pruning, i.e. weakest link pruning
## Tuning parameter alpha will penalize complexity as estimated by the number of terminal nodes
## As alpha increases from 0, pruning happens in a nested and predicatble way

## Determine the value of alpha using cross validation, cv.tree()

## Will pruning the tree improve performance?
## cv.tree() will run cross-validation analyses
## Cross validation will seek the minimize the deviance (dev) as a function of the tree size (i.e. cost-complexity factor, k)
?cv.tree
crosval.boston <- cv.tree(tree.boston)
plot(crosval.boston$size, crosval.boston$dev, type = 'b')

## Examining the plot, the deviance begins a minimum plateau at around k = 5
## Fewer splits results in more generalizable models, better performing on unseen (test) data
## Therefore, the prune.tree function can be used to optimize the k value
?prune.tree

prnd.boston <- prune.tree(tree.boston, best = 5)
plot(prnd.boston)
text(prnd.boston, pretty = 0)

summary(prnd.boston)
## Notice the pruned tree yields a greater deviance, and subsequently performs more poorly at predicting test data

## Make predictions (yhat) using the predict function
# Select as new data the Boston dataset without the rows (observations) from the training set
yhat <- predict(tree.boston, newdata = Boston[-train, ])
yhat1 <- predict(prnd.boston, newdata = Boston[-train, ])

## Create the test data set
# index the original data, and return those rows not included in 'train'(i.e. -train), and the target column
# Effectively, this selects the target column values not included in the training set
boston.test <- Boston[-train, 'medv']

plot(yhat, boston.test)
plot(yhat1, boston.test)
abline(0,1)
mse <- mean((yhat - boston.test)**2)
rmse <- sqrt(mse)

results[Model == 'Single Tree', 'MSE'] <- mse
results[Model == 'Single Tree', 'RMSE'] <- rmse

results

### The RMSE is 5.940
## Therefore, this regression tree predicts within $5,940 the true median home value

### Improving the Global Model by an Ensemble of Models
## Bagging
# Improved accuracy over predictions with a single tree
# Boostrapping by taking repeated samples from a single training data set, average all predictions (reg) / majority vote (class)
# i.e. create 'bags' of trees
# Grown deep, NOT pruned
library(randomForest)
set.seed(42)
bag.boston <- randomForest(medv ~ ., data = Boston,
                           subset = train,
                           mtry = 12, # try all 12 predictor variables at each split
                           importance = T) # calculate measures of variable importance
## Generate predictions using the bagged model
yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0,1)
mse.bag <- mean((yhat.bag - boston.test)**2)
rmse.bag <- sqrt(mse.bag)

results[Model == 'Bagged Trees', 'MSE'] <- mse.bag
results[Model == 'Bagged Trees', 'RMSE'] <- rmse.bag

## The RMSE of the bagged model is 4.833
# Therefore, the model has made predictions that are within $4,833 of the true value
# Note, this is an improvement from the single decision tree (RMSE = 5.940)

## Measures of Covariate Importance
importance(bag.boston)

## Two Measures
# % Inc MSE or % Dec Accuracy on out of bag samples when a given variable is permuted, i.e. by taking away the variable, how much accuracy is lost / gained?
# Inc in Node Purity or Dec in Impurity that results from splits over that variable
# Note: Node Purity is measured by the training RSS in regression trees, and by deviance for classification trees
varImpPlot(bag.boston)

### Random Forest Model
## Similar to bagging, but with smaller mtry values (i.e. using few covariates at each split)
## Default: p/3 covariates are used in a random forest regression trees
## Default: root p covariates in classification tree
set.seed(42)
rf.boston <- randomForest(medv ~ ., data = Boston,
                         subset = train,
                         mtry = 6,
                         importance = T)
yhat.rf <- predict(rf.boston, newdata = Boston[-train, ])
mse.rf <- mean((yhat.rf - boston.test)**2)
rmse.rf <- sqrt(mse.rf)

results[Model == 'Random Forest', 'MSE'] <- mse.rf
results[Model == 'Random Forest', 'RMSE'] <- rmse.rf

# This Random Forest model shows that the wealth of community (lstat) and the number of rooms (rm) are the most important variables
importance(rf.boston)
varImpPlot(rf.boston)

### Comparing the 3 models
results
## The random forest model minimizes the MSE and RMSE of the predictions when compared to the test data




