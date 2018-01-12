
# Wine Quality Investigation
# Date: 01/11/17
# Name: Rixing Xu
#
# Goal: Build a regression model to predict wine quality (dataset) using various machine
# learning techniques
#
# Multiple Step Process:
#   1) Descriptive Statistics
#   2) Feature Engineering
#   3) Predictions using Classification

###############################################
#     1.1 Load/ Install Required Packages     #
###############################################

install.packages("car")
install.packages("ggplot2")
install.packages("VGAM")
install.packages("class")
install.packages("randomForest")

library("car") # for vif factor
library("ggplot2") # for data visualization
library("VGAM") # logistic regression
library("class") # k neighbors
library("randomForest") # random forest 

###############################################
#     1.2  Read in Data (Red & White Wine)    #
###############################################

wine_r <- read.csv("C:/Wine/winequality-red.csv", head = TRUE, sep = ";") # reading in red wine data
wine_w <- read.csv("C:/Wine/winequality-white.csv", head = TRUE, sep = ";") # reading in white wine data

wineq <- read.table("C:/Wine/winequality-red.csv", head = TRUE, sep = ";")
attach(wineq)

predictors <- colnames(wineq)[1:length(colnames(wineq))-1] # our predictor variables
dep_var <- colnames(wineq)[length(colnames(wineq))]

str(wine_r) # looking at structure of the wine sets
str(wine_w)

summary(wine_r) # summary stats
summary(wine_w)

# To make life easier, we combine the two wine types
# first we create a new column for the wine type, then we stack the two dataframes
# 0 - Red, 1 - White
wine_r['type'] = 0
wine_w['type'] = 1

wine_all <- rbind(wine_r, wine_w)

##############################################
#     1.3    Data Visualization              #
##############################################
# visualizing the wine quality and predictors
par(mfrow = c(2,3))
for(p in predictors)
{
  plot(get(p), get('quality'), xlab = p, ylab = 'quality') 
  title(main = paste("Wine Quality vs ", p))
  Sys.sleep(2)
}

plottingFunction <- function(dataset, variables)
{
  par(mfrow = c(2,3))
  for(p in predictors)
  {
    plot(get(p), get('quality'), xlab = p, ylab = 'quality') 
    title(main = paste("Wine Quality vs ", p))
    Sys.sleep(2)
    boxplot(p, xlab = p)
    title(main =p)
  }
}

# correlation matrix
correlation_matrix = cor(wine_all)
write.csv(correlation_matrix, file = "C:/Wine/correlation_matrix.csv")

# function for getting the plots of the single variables 
#
# params: type_plot: linear(plots indep. var vs dep. var) , normal (plots the qq normality plot of the residuals of the model) 
# and homoscad (plots residuals vs indep. var)
plotting_func <- function(dataset, variables, type_plot, dep_var)
{
  for(var in variables){
    model <- lm(dep_var ~ 1, data = dataset)
    model <- update(model, as.formula(paste(".~.+", var)))
    pred <- predict(model)
    
    #par(mfrow = c(2,3))
    if(type_plot == 'linear'){
      plot(dataset[var], dataset[dep_var], main = paste("Wine Quality vs ", var))
      Sys.sleep(2)
    }
    else if(type_plot == 'normal'){
      qqnorm(resid(model), main = var)
      Sys.sleep(2)
    }
    else if(type_plot == 'homoscad')
    {
      plot(dataset[var], resid(model), main = paste("Residuals vs ", var))
      Sys.sleep(2)
    }
    else{
      print("Try again")
    }
  }
}
plotting_func(wine_r, predictors, 'normal', dep_var)

##############################################
#     1.4    Summary Statistics              #
##############################################
wine_stat_init <- data.frame()

for(v in predictors)
{
  model<- lm(quality ~ 1)
  model <- update(model, as.formula(paste('.~. +', v)))
  
  wine_stat_init <- rbind(wine_stat_init, c(summary(model)$r.squared, cor(wine_all[v], wine_all[dep_var]), anova(model)$`Pr(>F)`))
}
colnames(wine_stat_init) <- c("r_sq", "corr", "p_value", "r_score")
rownames(wine_stat_init) <- predictors

wine_stat_init <- wine_stat_init[colSums(!is.na(wine_stat_init)) > 0]

write.csv(wine_stat_init, file = 'C:/Wine/wine_stat_init.csv')

##############################################
#     2.1   Fill Missing Values              #
##############################################
for(p in predictors)
{
  if(colSums(is.na(wine_all[p])) > 0)
  {
    mean = as.numeric(sub('.*:', '', summary(wine_all[p])[4]))
    wine_all[(is.na(wine_all[p]))] = mean
  }
}
# no missing values for our datasets

##############################################
#     2.2   Outliers Detection               #
##############################################
# plotting box plots 
par(mfrow = c(2,3))
for (p in predictors)
{
  boxplot(get(p), main = p)
  Sys.sleep(2)
}

# checks the cooks distance of each outlier point and then replaces it with the mean of 
# that dataset if the point is not influential
outlierRemoval <- function(dataset, variables)
{
  par(mfrow = c(2,3))
  for (p in variables)
  {
    # manually calculating extreme outliers
    lower = quantile(as.vector(as.matrix(dataset[p])))[2]
    upper = quantile(as.vector(as.matrix(dataset[p])))[4]
    iqr = upper - lower
    
    outliers <- (dataset[p])[dataset[p] > ((iqr*3) + upper) | dataset[p] < (lower - (iqr*3))]
    outliers_loc <- which(dataset[p] > ((iqr*3) + upper) | dataset[p] < (lower - (iqr*3))) # outlier locations
    
    mod <- lm(quality ~ 1, data = dataset)
    mod <- update(mod, as.formula(paste('.~. +', p)))
    cooksd <- cooks.distance(mod) # saves cook's distance of each data point
    plot(cooksd, pch = '*', cex = 1, main = paste(p, 'Influencial Indexes'))
    abline(h = 4/(nrow(dataset)), col = 'red')
    Sys.sleep(1)
    for(i in outliers_loc)
    {
      if(cooksd[i] > 4/(nrow(dataset)))
      {
        mean = as.numeric(sub('.*:', '', summary(wine_all[p])[4]))
        dataset[p][i,] = mean  
      }
    }
  }
}
outlierRemoval(wine_all, predictors)

##############################################
#     2.3   Check for Duplicates             #
##############################################

wine_all <- wine_all[!duplicated(wine_all),] # removes duplicates if any

###########################################################
#     2.4   Scale Transform Variables(Regression only)    #
###########################################################
# looking to transform data to the same scale:
for(p in predictors)
{
  wine_all[p] <- scale(wine_all[p])
}

##############################################
#     2.5   Check for Multicollinearity      #
##############################################
vif_matrix = vif(wine_all[predictors])
write.csv(vif_matrix, file = "C:/Wine/vif_matrix.csv")

# remove highly correlated variables > VIF of 10
predictors_new <- as.vector(as.matrix(((vif_matrix[vif_matrix[,'VIF'] < 10.0,])['Variables'])))

##############################################
#     3.1   Split Train/Test Sets            #
##############################################

wine_all$id <- seq.int(nrow(wine_all)) 
wine_all_new <- wine_all[predictors_new]
wine_all_new['quality'] <- wine_all['quality']
w_train = wine_all[which(wine_all$id <=nrow(wine_all) /2),]
w_test = wine_all[which(wine_all$id > nrow(wine_all) /2),]

model <- vglm(quality ~., family = multinomial, data = w_train)
summary(model)
prob <- predict(model, w_test, type = 'response')
y_pred <- apply(prob, 1, which.max)
table(y_pred, w_test$quality)

print(mean(as.vector(as.matrix(y_pred)) == as.vector(as.matrix(w_test['quality']))))

##############################################
#     3.2   Linear Regression                #
##############################################
model <- lm(quality ~ 1, data = w_train)

for (p in predictors_new)
{
  model = update(model, as.formula(paste('.~. +', p)))
}

y_pred <- predict(model, w_test)

for(i in 1:length(y_pred)) # rounding the predictions to nearest integrer
{
  y_pred[[i]] <- round(y_pred[[i]])
}

print(mean(y_pred == w_test['quality'])) # accuracy of 0.518


##############################################
#     3.3       K Neighbors                  #
##############################################
wine_stats <- as.vector(as.matrix(w_train['quality'])) # out target variable

pred <- knn(train = w_train, test = w_test, cl = wine_stats, k = 3)

mean(pred == as.vector(as.matrix(w_test['quality']))) # accuracy of 0.466

##############################################
#     3.4      Random Forest                 #
##############################################
model <- randomForest(quality ~., data = w_train)
pred <- predict(model, newdata = w_test)

for(i in 1:length(pred))
{
  pred[[i]] <- round(pred[[i]])
}

table(pred, w_test$quality)
print(mean(pred == w_test['quality'])) # accuracy of 0.525
