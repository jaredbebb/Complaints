library(tm)
library(qdap)
library(wordcloud)

library(RTextTools)
library(caret)

memory.limit(size=10000)

s = file.choose()
surveys <- read.csv(s)

surveys
 
#removing return characters
surveys$Comments <- stringr::str_replace_all(surveys$Comments, "[\r\n]", "")
surveys$Comments <- tolower(surveys$Comments)
surveys$Comments <- removePunctuation(surveys$Comments)
 
#commented out removeWords because several instances have only one word, and it's a stopword
surveys$Comments <- removeWords(surveys$Comments, stopwords("english"))
surveys$Comments <- stripWhitespace(surveys$Comments)
surveys$Comments <- trimws(surveys$Comments)

#stemDocument uses Porter's stemmer algorithm
surveys$Comments <- stemDocument(surveys$Comments)
 
frequent_terms <- qdap::freq_terms(surveys$Comments,30)
plot(frequent_terms)
surveys$Comments
 
#changing rows with emptpy strings to NA, need to remove these empty rows
surveys$Comments[surveys$text== "" ]<- NA
surveys$Comments
 
frequent_terms <- qdap::freq_terms(surveys$Comments,30)
plot(frequent_terms)

#some survey Reasons have different uppcase and lowercase letters
surveys$Reason <-tolower(surveys$Reason)

#replace space and slash with underscore
surveys$Reason
surveys$Reason <- stringr::str_replace_all(surveys$Reason, "[ /]", "_")
surveys$Reason
#some survey reasons are just "n/a" or "no". Is really a null label
surveys$Reason[surveys$Reason== "na" ]<- NA
surveys$Reason[surveys$Reason== "n_a" ]<- NA
surveys$Reason[surveys$Reason== "no" ]<- NA

#change unlabeled data to NA
surveys$Reason <- as.character(surveys$Reason)
surveys$Reason[surveys$Reason ==""] <- NA
surveys$Reason<- as.factor(surveys$Reason)
surveys$Reason

#removing NA values from surveys data frame
surveys <- na.omit(surveys)
surveys
 
frequent_class <- qdap::freq_terms(surveys$Reason,30)
frequent_class
plot(frequent_class)

library(quanteda)
#uses n-gram model for document frequency matrix
surveys_dfm <- dfm(surveys$Comments, ngrams=1:4, verbose = FALSE)

#only include features that show up in 2 or more surveys
surveys_dfm <- dfm_trim(surveys_dfm, min_count = 4)
surveys_dfm

dfm_m <- as.matrix(surveys_dfm)
inspect(dfm_m)

#get reason label
d_reason <- surveys$Reason
d_reason

#row_count <- nrow(d_reason) 
row_count <- length(d_reason)
 
#train/test 80/20 of data
train_container <- create_container(dfm_m,labels=d_reason, trainSize = 1:(row_count*.80),
                                     virgin = FALSE)
test_container <- create_container(dfm_m,labels=d_reason, trainSize = (row_count*.80):row_count,
                                    virgin = FALSE)
test_nps_category <- d_reason[(row_count*.80):row_count]

# train linear SVM model
# Accuracy :   
svm_linear_model <- train_model(train_container, "SVM", kernel="linear", cost=1)
slm<-classify_model(test_container,svm_linear_model)
slm_results<-slm$SVM_LABEL
confusionMatrix(slm_results,test_nps_category)
 
#train tree model
# Accuracy :   
tree_linear_model <- train_model(train_container, "TREE", kernal="linear", cost=1)
tlm<- classify_model(test_container,tree_linear_model)
tlm_results<-tlm$TREE_LABEL
confusionMatrix(tlm_results,test_nps_category)
 
#train randomforest model
#Accuracy :   
randomf_linear_model <- train_model(train_container, "RF", kernal="linear", cost=1)
rflm <- classify_model(test_container, randomf_linear_model)
rflm_results <- rflm$FORESTS_LABEL
confusionMatrix(rflm_results, test_nps_category)

#train neural network model
#Accuracy : 
nnet_linear_model <- train_model(train_container, "NNET", kernal="linear", cost=1)
nlm <- classify_model(test_container,nnet_linear_model)
nlm_results <- nlm$NNETWORK_LABEL
confusionMatrix(nlm_results, test_nps_category)
