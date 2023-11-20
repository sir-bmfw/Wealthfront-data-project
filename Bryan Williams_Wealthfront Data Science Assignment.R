library(ROCR)

### Given that the assignment asks for a binary outcome (i.e. "good" or "bad" loans),
### I'm going with a logistic regression for a model. I'll approach this by
### pulling out the completed loans for training/testing, and then run the model
### on the in-progress loans to predict what they'll do. 

# read in loan data
loans_raw <- read.csv("Data Pulls/loan_data.csv")
loans_raw$earliest_cr_line <- as.Date(loans_raw$earliest_cr_line, format="%Y-%m-%d")
# remove blank rows
loans_raw <- subset(loans_raw,loan_status!="")
# change qualitative dimensions to factors
loans_raw <- as.data.frame(unclass(loans_raw),stringsAsFactors = TRUE)
# let's remove the chaff at the end of dataset
loans_clean <- data.frame(loans_raw[,c(1:23)],row.names=1)
# finally, let's calculate a 'pct_of_income' that takes the percentage of the 
# 'funded_amnt' relative to 'annual_inc' as a simple coverage dimension 
loans_clean$pct_of_inc <- loans_clean$funded_amnt / loans_clean$annual_inc

# 'loan_status' consists of seven different outcomes. Three of these are end results,
# with 'Fully Paid', 'Charged Off', and 'Default' being end statuses. We will create
# a new dimension where we cluster these together in order to help us with our model
loans_clean$result <- ifelse(loans_clean$loan_status=="Fully Paid","good",
                             ifelse(loans_clean$loan_status=="Charged Off","bad",
                                    ifelse(loans_clean$loan_status=="Default","bad","in progress")))

# we'll change the good/bad label to a binomial for the model
loans_end <- subset(loans_clean,result=="good"|result=="bad")
loans_end$result <- ifelse(loans_end$result=="good",1,0)

# we'll use the loans_end data to train and test our model, and then use
# it on the 'in progress' loans to see which side they're likely to fall on!

sapply(loans_end,function(x) sum(is.na(x)))
# 'mths_since_last_delinq' has a bunch of missing values, so we will not use it
# also removing 'out_prncp', 'loan_status', and the totals at the end
loans_end <- subset(loans_end,select=-c(mths_since_last_delinq,out_prncp,
                                        loan_status,
                                        total_pymnt,total_rec_prncp,
                                        total_rec_int))

# split up train and test data; we'll go 70/30 on it
end_train <- sample_frac(loans_end, 0.7)
sid <- as.numeric(rownames(end_train))
end_test <- loans_end[-sid,]

# logistic regression model on the 'result' binomial
model <- glm(result ~.,family=binomial(link='logit'),data=end_train)
summary(model)

# check ANOVA with Chi-square test 
anova(model, test="Chisq")
# 'funded_amnt', 'int_rate', 'emp_length','annual_inc','purpose','dti','total_acc',
# 'pct_of_inc' all have significance

# testing model accuracy on the test data
fitted.results <- predict(model,newdata=end_test,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misError <- mean(fitted.results != end_test$result)
print(paste('Model Accuracy = ',1-misError))

# plot False Positive Rate
p <- predict(model,newdata=end_test,type='response')
pr <- prediction(p, end_test$result)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

# calculate area under curve
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

### redo, removing 'addr_state' in order to run against live loan data
### (there is no VT in the finished loan data and 'addr_state' is insignificant,
### so it'll just be easier to remove it than work around it)
loans_end <- subset(loans_end,select=-c(addr_state))

end_train <- sample_frac(loans_end, 0.7)
sid <- as.numeric(rownames(end_train))
end_test <- loans_end[-sid,]

model <- glm(result ~.,family=binomial(link='logit'),data=end_train)
summary(model)

anova(model, test="Chisq")

fitted.results <- predict(model,newdata=end_test,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misError <- mean(fitted.results != end_test$result)
print(paste('Model Accuracy = ',1-misError))
# 85.6%, even better!

# read in live loan data, removing two rows due to aberrant 'home_ownership' values
loans_live <- subset(loans_clean,result=="in progress"&(home_ownership!="NONE"&home_ownership!="OTHER"))
loans_live <- subset(loans_live,select=-c(addr_state))

# predict loan success!
loans_live$logit_predict <- predict(model,newdata=loans_live,type="response")
loans_live$prediction <- ifelse(loans_live$logit_predict > 0.5, "good","bad")

# generate some summary tables
total <- data.frame(table(loans_live['prediction']))
total
good_tot <- data.frame(table(subset(loans_live,prediction=="good")['loan_status']))
good_tot <- good_tot[c(2,5:7),]
good_tot
bad_tot <- data.frame(table(subset(loans_live,prediction=="bad")['loan_status']))
bad_tot <- bad_tot[c(2,5:7),]
bad_tot
bad_pct <- bad_tot
bad_pct$Freq <- bad_tot$Freq / (bad_tot$Freq + good_tot$Freq)
bad_pct

