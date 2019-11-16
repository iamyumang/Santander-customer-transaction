x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees', "plyr","dplyr", "rpart", "usdm", "DataCombine", "sp", "raster", "usdm")

install.packages(x)
lapply(x, require, character.only = TRUE)

install.packages("party")
library(party)
library(data.table)
library(dplyr)
library(lightgbm)

rm(list = ls())
getwd()
train = read.csv("train_san.csv", header = TRUE, sep = ",")
df_test = read.csv("test_san.csv", header = TRUE, sep = ",")
class(train$target)
sum(is.na(train))
train$target = as.factor(train$target)
class(train$target)
summary(train)
head(train)
dim(train)
str(train)
names(train)

###########################Check for missing values###########################
sum(is.na(train))
table(train$target)

#after extracting the unique values in target column, we can clearly say that the data is imbalanced
length(which(train$target == 1))/length(train$target) *100

# 10.049% of target values is equal to 1


par(mar=c(1,1,1,1))
par(mfrow=c(2,2))
for (col in 3:ncol(train)) {
  hist(train[,col])
}


for (col in 3:ncol(train)) {
  boxplot(train[,col])
}


df_train = copy(train)

df_train$ID_code = NULL
class(df_train$target)


for(i in 2:ncol(df_train)){
  
  if(class(df_train[,i]) == 'factor'){
    
    df_train[,i] = factor(df_train[,i], labels=(2:length(levels(factor(df_train[,i])))))
    
  }
}


############################################Outlier Analysis#############################################
# ## BoxPlots - Distribution and Outlier Check
numeric_index = sapply(df_train,is.numeric) #selecting only numeric

numeric_data = df_train[,numeric_index]

cnames = colnames(numeric_data)


 # #Remove outliers using boxplot method
 #df = train
 #train = df


fun = function(x){
  quantiles = quantile( x, c(.05, .95 ) )
  x[ x < quantiles[1] ] = quantiles[1]
  x[ x > quantiles[2] ] = quantiles[2]
  x
}

df_train[,cnames] = lapply(df_train[,cnames], fun)

 for (col in 2:ncol(df_train)) {
    boxplot(df_train[,col])
 }

sum(is.na(df_train))

df_train_final = copy(df_train)

## Correlation Plot 
#corrgram(df_train[,numeric_index], order = F,
#         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#correlation = cor(df_train[,c(2:201)])
#class(correlation)

aov(df_train$target~df_train[,cnames])


# for all categorical variables

formula = as.formula(paste0("cbind(", paste(names(df_train)[-1], collapse = ","), ") ~ target"))

Anova_test_result = aov(formula, data=df_train)
summary(Anova_test_result)

class(Anova_test_result)

#since the dataset is imbalance, we have to do oversampling or undersampling to overcome this problem
 
set.seed(1234)
d_train.index = createDataPartition(df_train$target, p = .75, list = FALSE)
d_train = df_train[ d_train.index,]
test  = df_train[-d_train.index,]


table(d_train$target)
over = ovun.sample(target~., data= d_train, method = "over", N = 269854)$data
table(over$target) 

under = ovun.sample(target~., data= d_train, method = "under", N = 30148)$data
table(under$target)

#*******************************predict using Logistic Regression(for upsampling and downsamplng both)********************

LR_under = glm(target ~.,data = under, family = binomial)

predict_LR_under = predict(LR_under, test, type="response")

probabilities = LR_under %>% predict(test, type = "response")

predict_LR_under = ifelse(probabilities > 0.5, "1", "0")

predict_LR_under = as.factor(predict_LR_under)

confusionMatrix(predict_LR_under, test$target,positive = "1")


#---------------------------------------------------------------------------------


LR_over = glm(target ~.,data = over, family = binomial)

predict_LR_over = predict(LR_over, test, type="response")

probabilities = LR_over %>% predict(test, type = "response")

predict_LR_over = ifelse(probabilities > 0.5, "1", "0")

predict_LR_over = as.factor(predict_LR_over)

confusionMatrix(predict_LR_over, test$target,positive = "1")





#*******************************predict using decision tree(for upsampling and downsamplng both)********************



DT_under = ctree(target~., data = under)
confusionMatrix(predict(DT_under,test), test$target,positive = "1")
#--------------------------------------------------------------------------------
#DT_over = ctree(target ~ . , data = over)
#confusionMatrix(predict(DT_over,test), test$target,positive = "1")




#*******************************predict using random forest(for upsampling and downsamplng both)********************


rf_under = randomForest(target~. , data = under , importance = TRUE , ntree = 320)
confusionMatrix(predict(rf_under,test), test$target,positive = "1")

#---------------------------------------------------------------------------------

#rf_over = randomForest(target~. , data = over , importance = TRUE , ntree = 300)
#confusionMatrix(predict(rf_over,test), test$target,positive = "1")



#**************************************************************************************
#*************************************************************************************************
#***********************manipulating test data for prediction****************************************************

getwd()

df_test1 = copy(df_test)
df_test1$ID_code = NULL
head(df_test1)



###############################predict for test data#######################################
predict_LR_test = predict(LR_over, df_test1, type="response")

probabilities_test = LR_over %>% predict(df_test1, type = "response")

predict_LR_test = ifelse(probabilities_test > 0.5, "1", "0")

df_test1$predicted_target = predict_LR_test

test_final = cbind(df_test, df_test1$predicted_target)
head(test_final)
table(test_final$`df_test1$predicted_target`)

#write.csv(test_final,file = file.choose(new = T))
