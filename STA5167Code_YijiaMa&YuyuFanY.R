#R code used in STA5167 Final Project
#Predicting footballer's market value with multiple linear regression
#Yuyu Fan & Yijia Ma 

### The package used in our models
library(eeptools)
library(caTools)
library(corrplot)
library(RColorBrewer)
library(MASS)
library(car)
library(olsrr)
library(ggplot2)
library(GGally)
library(plyr)
library(alr4)
library(randomForest)
library(lars)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(rsample)     # data splitting 
library(dplyr)       # data wrangling
library(rpart)       # performing regression trees
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(caret)       # bagging
library(randomForestExplainer)
library(tree)
library(party)
library(Hmisc)


### Data Preprocessing
train<-read.csv('train.csv',sep = ',')

## get age for players
train$birth_date = as.Date(train$birth_date, format = "%m/%d/%Y")
train$age = age_calc(train$birth_date,enddate = as.Date("2020-03-28"), units = "years", precise = TRUE)

## get the highest score in all positions
position = c('rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk')
train[,"best_pos"]<-apply(train[,position],1,max, na.rm=TRUE) 

## calculate BMI for all players
train[,'BMI'] = 10000. * train$weight_kg / (train$height_cm ** 2)

## change caltogorical data to factor
train$work_rate_att<-factor(train$work_rate_att)
train$work_rate_def<-factor(train$work_rate_def)

## Determine if the player is goalkeeper
train['is_gk'] = ifelse(is.na(train$gk),0,1)

## drop replicate info columns
drops = c('id','birth_date','rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk')
train = train[ , !(names(train) %in% drops)]

## split training and test set in train
set.seed(123)
split = sample.split(train$y,SplitRatio = 0.8)
training_set = subset(train, split ==TRUE)
test_set = subset(train, split ==FALSE)


### Multiple linear regression
## scatterplots
for (i in colnames(training_set)){
  plt<-ggplot(aes_string(x=i,y=training_set$y),data = training_set)+
    scale_y_continuous(trans='log2')+
    geom_point(color="#B20000", size=4, alpha=0.5) +
    geom_smooth(method=lm, alpha=0.25, color="black", fill="black")
  print(plt)
}

#subset data
is_gk = subset(training_set, is_gk==1)
is_gk = subset(is_gk , select = -c(is_gk))
not_gk = subset(training_set, is_gk==0)
not_gk = subset(not_gk, select = -c(is_gk))

##model fitting 
## Analyze the correlation between y and other predictors
fit1.1<-lm(y~.,data = subset(is_gk,select=-c(skill_moves,work_rate_att,work_rate_def)))
summary(fit1.1)
ncvTest(fit1.1)
plot(fit1.1) 

fit1.2<-lm(y~.,data = not_gk)
summary(fit1.2)
ncvTest(fit1.2)
plot(fit1.2) 


##transform the response variable
bc1.1=boxcox(fit1.1,lambda = seq(-3,3))
best.lam.1.1 = bc1.1$x[which(bc1.1$y == max(bc1.1$y))]

bc1.2=boxcox(fit1.2,lambda = seq(-3,3))
best.lam.1.2 = bc1.2$x[which(bc1.2$y == max(bc1.2$y))]


#fit the model after transformed
fit2.1<-lm(log(y)~.,data = subset(is_gk,select=-c(skill_moves,work_rate_att,work_rate_def)))
summary(fit2.1)
plot(fit2.1)
ncvTest(fit2.1)

fit2.2<-lm(log(y)~.,data = not_gk)
summary(fit2.2)
ncvTest(fit2.2)
plot(fit2.2) 

#inverse response plot
invResPlot(fit2.1)
invResPlot(fit2.2)



#plot the correlation matrix
data=subset(is_gk,select=-c(skill_moves,work_rate_att,work_rate_def))
res2 <- rcorr(as.matrix(data))
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}
c=flattenCorrMatrix(res2$r, res2$P)
c


##variable selection 
# Stepwise Regression
fit1 <- lm(log(y)~.,data=subset(is_gk,select=-c(skill_moves,work_rate_att,work_rate_def)))
step1 <- stepAIC(fit1, direction="both")
step1$anova # display results

fit2 <- lm(log(y)~.,data=not_gk)
step2 <- stepAIC(fit2, direction="both")
step2$anova # display results


#plot the variable selection process
fit1 <- lm(log(y)~.,data=subset(is_gk,select=-c(skill_moves,work_rate_att,work_rate_def)))
k <- ols_step_both_aic(fit1)
plot(k)
k

fit2 <- lm(log(y)~.,data=not_gk)
k2 <- ols_step_both_aic(fit2)
plot(k2)
k2
k2$fit2


#transform the predictors
summary(b1 <- powerTransform(cbind(log(y), league , potential, international_reputation , preferred_foot, short_passing , volleys, dribbling , agility , strength ,  vision , penalties , age ,best_pos) ~ 1,family="bcPower",data=is_gk))
summary(b1 <- powerTransform(cbind(log(y), league , potential, international_reputation , preferred_foot, short_passing , volleys, dribbling , agility , strength ,  vision , penalties , age ,best_pos) ~ 1,data=is_gk))
summary(b2 <- powerTransform(cbind(y , nationality , potential , international_reputation ,
                                   skill_moves , work_rate_def , crossing , finishing , heading_accuracy , volleys , dribbling , free_kick_accuracy, long_passing , ball_control ,sprint_speed , agility , reactions , balance , 
                                   stamina , strength,long_shots ,positioning , penalties,  marking, gk_positioning , age , best_pos) ~ 1,data=not_gk))

testTransform(b1,c(0,1,1,-15,-12,0.5,0.5,0.5,0.5,2,1,0,-1,1)) ##149.9219
testTransform(b1,c(0,1,1,0,0,0.5,0.5,0,0,2,1,0,-1,1)) ##6135.647	
testTransform(b1,c(0,1,1,1,1,1,1,0,0,1,1,0,-1,1))##7957.942
testTransform(b1,c(0,1,1,1,1,1,1,0,0,1,1,0,1,1))##8249.599
testTransform(b1,c(0,1,1,1,1,1,1,1,1,1,1,1,1,1))##8636.437

testTransform(b2,c(0,1,1,-13,-2,3,2,2,2,1,2,0.5,2,3,2,2,2,2,2,2,1,2,1,1,1,-1,2))#2606.804
testTransform(b2,c(0,1,1,0,0,1,1,1,1,1,1,0.5,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1)) #42320.17
testTransform(b2,c(0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1)) #42808


#caculate relative importance
calc.relimp(fit3.1, type = c("lmg"), rela = TRUE) 
bootresults<-boot.relimp(fit3.1, b=1000) 
ci<-booteval.relimp(bootresults, norank=T)
plot(ci)

calc.relimp(fit3.2, type = c("lmg"), rela = TRUE) 
bootresults2<-boot.relimp(fit3.2, b=10) 
ci2<-booteval.relimp(bootresults2, norank=T)
plot(ci2)



##final model 
fit3.1.1<-lm(log(y) ~ league + potential + international_reputation + preferred_foot + 
               short_passing + volleys + dribbling + agility + strength + 
               vision + penalties + age+best_pos,data = is_gk)#final model

fit3.2.1<-lm(log(y) ~  nationality + potential + international_reputation + 
               skill_moves + work_rate_def + crossing + finishing + heading_accuracy + 
               volleys + dribbling + free_kick_accuracy + long_passing + 
               ball_control + sprint_speed + agility + reactions + balance + 
               stamina + strength + long_shots + positioning + penalties + 
               marking + gk_positioning + age + best_pos,data = not_gk)#final model

summary(fit3.1.1)
summary(fit3.2.1)


#residual diagnostics
plot(fit3.1.1)
plot(fit3.2.1)


ncvTest(fit3.1.1)
ncvTest(fit3.2.1)


shapiro.test(fit3.1.1$residuals)
shapiro.test(fit3.2.1$residuals)


residualPlots(fit3.1.1)
residualPlots(fit3.2.1)


## MAE
test_set$ypred = 0
test_set[test_set[ ,'is_gk'] == 0, ]$ypred = predict(fit3.1.1,test_set[test_set[ ,'is_gk'] == 0, ])
test_set[test_set[ ,'is_gk'] == 1, ]$ypred = predict(fit3.2.1,test_set[test_set[ ,'is_gk'] == 1, ])
test_set$y_pred = exp(test_set$ypred)
MAE(test_set$y_pred,test_set$y)


## Extra work not in report
#add weights
fit3.1.1<-lm(log(y) ~ league + potential + international_reputation + preferred_foot + 
               short_passing + volleys + dribbling + agility + strength + 
               vision + penalties + age+best_pos,weights = 1/international_reputation^2,data = is_gk)

fit3.2.1<-lm(log(y) ~  nationality + potential + international_reputation + 
               skill_moves + work_rate_def + crossing + finishing + heading_accuracy + 
               volleys + dribbling + free_kick_accuracy + long_passing + 
               ball_control + sprint_speed + agility + reactions + balance + 
               stamina + strength + long_shots + positioning + penalties + 
               marking + gk_positioning + age + best_pos,weights = 1/international_reputation^2,data = not_gk)
# cross validation
model_gk <- train(log(y) ~., data = training_set[training_set[ ,'is_gk'] == 1, ], method = "lm",
                  trControl = train.control)

model_notgk <- train(log(y) ~., data = training_set[training_set[ ,'is_gk'] == 0, ] , method = "lm",
                     trControl = train.control)
print(model)
test_set$ypredcv = 0
test_set[test_set[ ,'is_gk'] == 0, ]$ypredcv = predict(model_notgk,test_set[test_set[ ,'is_gk'] == 0, ])
test_set[test_set[ ,'is_gk'] == 1, ]$ypredcv = predict(model_gk,test_set[test_set[ ,'is_gk'] == 1, ])
MAE(exp(test_set$ypredcv),test_set$y)





### Random Forest
### Random forest with all predictors into two models
set.seed(123)
fit1.2<-randomForest(y~., data = training_set[training_set[ ,'is_gk'] == 0, ])
set.seed(123)
fit1.3 <-randomForest(y~., data = training_set[training_set[ ,'is_gk'] == 1, ])
test_set$ypred = 0
test_set[test_set[ ,'is_gk'] == 0, ]$ypred = predict(fit1.2,test_set[test_set[ ,'is_gk'] == 0, ])
test_set[test_set[ ,'is_gk'] == 1, ]$ypred = predict(fit1.3,test_set[test_set[ ,'is_gk'] == 1, ])
MAE(test_set$ypred,test_set$y)

### Random forest with predictors returned in the MLR into two models
set.seed(123)
fit1.2.2<-randomForest(y~nationality + potential + international_reputation + 
                         skill_moves + work_rate_def + crossing + finishing + heading_accuracy + 
                         volleys + dribbling + free_kick_accuracy + long_passing + 
                         ball_control + sprint_speed + agility + reactions + balance + 
                         stamina + strength + long_shots + positioning + penalties + 
                         marking + gk_positioning + age + best_pos, data = training_set[training_set[ ,'is_gk'] == 0, ])
set.seed(123)
fit1.3.2 <-randomForest(y~league + potential + international_reputation + preferred_foot + 
                          short_passing + volleys + dribbling + agility + strength + 
                          vision + penalties + age+best_pos, data = training_set[training_set[ ,'is_gk'] == 1, ])
test_set$y_pred = 0
test_set[test_set[ ,'is_gk'] == 0, ]$y_pred = predict(fit1.2.2,test_set[test_set[ ,'is_gk'] == 0, ])
test_set[test_set[ ,'is_gk'] == 1, ]$y_pred = predict(fit1.3.2,test_set[test_set[ ,'is_gk'] == 1, ])
MAE(test_set$y_pred,test_set$y)

### Random forest with all predictors into one model
set.seed(123)
fit1.4<-randomForest(y~., data = training_set)
test_set$y_pred_comp = 0
test_set$y_pred_comp = predict(fit1.4,test_set)
MAE(test_set$y_pred_comp,test_set$y)

### Important varibles we get from random forest

varImpPlot(fit1.2, sort=T, n.var = 10, main = "Top 10 -Variable Importance")
importance(fit1.2)
varUsed(fit1.2)

varImpPlot(fit1.3, sort=T, n.var = 10, main = "Top 10 -Variable Importance")
importance(fit1.3)
varUsed(fit1.3)


varImpPlot(fit1.2.2, sort=T, n.var = 10, main = "Top 10 -Variable Importance")
importance(fit1.2.2)
varUsed(fit1.2.2)

varImpPlot(fit1.3.2, sort=T, n.var = 10, main = "Top 10 -Variable Importance")
importance(fit1.3.2)
varUsed(fit1.3.2)

varImpPlot(fit1.4, sort=T, n.var = 10, main = "Top 10 -Variable Importance")
importance(fit1.4)
varUsed(fit1.4)



## Decision Tree
m1<-rpart(
  formula = y ~ .,
  data    = training_set[training_set[ ,'is_gk'] == 0, ],
  method  = "anova",
  control = list(minsplit = 17, maxdepth = 14, cp = 0)
  #cp = 0.0
)
m2<-rpart(
  formula = y ~ .,
  data    = training_set[training_set[ ,'is_gk'] == 1, ],
  method  = "anova",
  control = list(minsplit = 17, maxdepth = 15, cp = 0)
  #cp = 0.0
)
m3<-rpart(
  formula = y ~ .,
  data    = training_set,
  method  = "anova",
  control = list(minsplit = 15, maxdepth = 12, cp = 0)
  #cp = 0.0
)

test_set$ypred_tree = 0
test_set[test_set[ ,'is_gk'] == 0, ]$ypred_tree = predict(m1,test_set[test_set[ ,'is_gk'] == 0, ],type="vector")
test_set[test_set[ ,'is_gk'] == 1, ]$ypred_tree = predict(m2,test_set[test_set[ ,'is_gk'] == 1, ],type="vector")

test_set$ypred_tree_comp = 0
test_set$ypred_tree_comp = predict(m3,test_set,type="vector")

MAE(test_set$ypred_tree,test_set$y)
MAE(test_set$ypred_tree_comp,test_set$y)









