## Load decision tree library
library(rpart)
## Load random forest library
library(randomForest)
## Load k-nearest neighbours library
library(class)
## Load cross validation library
library(cvTools)
## Load support vector machine library
library(e1071)
## Load RUnit library
library(RUnit)
## Load graphics libraries
library(rattle)
library(rpart.plot)
library(RColorBrewer)

modelPrediction <- function(model_name, params, trData, teData){
    model_prediction <- NULL
    ## Create baseline prediction
    if (model_name == "baseline") {
        model_prediction <- ifelse(teData$Sex == "female", 1, 0)
    } 
    ## Create decision tree prediction
    else if (model_name == "dTree") {
        dTree <- rpart(
            Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare,
            data=trData,
            method ="class",
            parms=list(split="information"),
            control=params
            )
        model_prediction <- predict(dTree, teData, "class")
        # fancyRpartPlot(dTree)
    } 
    ## Create random forest
    else if (model_name == "forest") {
        forest <- randomForest(
            as.factor(Survived) ~ 
                Pclass + Sex + Age + Fare + Embarked + FamilySize + Child,
            data=trData,
            importance=TRUE,
            ntree=params[1],
            mtry=params[2])
        model_prediction <- predict(forest, teData)
        # varImpPlot(forest)
    }
    ## Create k-nearest neighbours
    else if(model_name == "knn") {
        features <- c('nAge', 'nFamilySize', 'nFare', 'nPclass', 'nSex')
        model_prediction <- knn(train=trData[features],
                                test=teData[features],
                                cl=trData$Survived, k=31)
    }
    ## Create support vector machine model
    else if(model_name == "svm") {
        features <- c('nAge', 'nFamilySize', 'nFare', 'nPclass', 'nSex')
        svm_model <- svm(trData[features], trData$Survived)
        model_prediction <- round(predict(svm_model, teData[features]))
    }
    ## Verify data size
    checkEquals(nrow(data.frame(model_prediction)), nrow(teData))
    return(model_prediction)
}

crossValidation <- function(model_name, params, data, num_folds) {
    folds <- cvFolds(nrow(data), K=num_folds)
    errors <- NULL

    for (i in 1:num_folds) {
        trData <- data[folds$subsets[folds$which != i],]
        teData <- data[folds$subsets[folds$which == i],]
        model_prediction <- data.frame(modelPrediction(
            model_name=model_name,
            params=params,
            trData=trData,
            teData=teData))
        errors[i] <- sum(model_prediction == teData$Survived) / nrow(teData)
    }
    return(mean(errors))
}

normalise <- function(x) {
    return( (x - min(x)) / (max(x) - min(x)) )
}

extractData <- function() {
    train_url <- "data/train.csv"
    train <- read.csv(train_url)
    test_url <- "data/test.csv"
    test <- read.csv(test_url)

    ## Clean data
    all_data <- rbind(train[,names(train) != "Survived"], test)
    ## Replace missing embarked values with majority
    all_data$Embarked[is.na(all_data$Embarked)] <- "S"
    ## Factorise embarkment codes
    all_data$Embarked <- factor(all_data$Embarked)
    ## Replace missing fare values with median
    all_data$Fare[is.na(all_data$Fare)] <- median(all_data$Fare, na.rm = TRUE)
    ## Replace missing age values with median
    all_data$Age[is.na(all_data$Age)] <- median(all_data$Age, na.rm = TRUE)

    ## Derive new features
    all_data$Child <- factor(ifelse(all_data$Age >= 18, "adult", "child"))
    all_data$FamilySize <- all_data$SibSp + all_data$Parch + 1

    # all_data$Title <- "None"
    # all_data$Title[grep("Lady|Sir|Col", all_data$Name)] <- "Titled"
    # all_data$Title <- factor(all_data$Title)

    # all_data$Deck <- "None"
    # all_data$Deck[grep("A", all_data$Cabin)] <- "A"
    # all_data$Deck[grep("B", all_data$Cabin)] <- "B"
    # all_data$Deck[grep("C", all_data$Cabin)] <- "C"
    # all_data$Deck[grep("D", all_data$Cabin)] <- "D"
    # all_data$Deck[grep("E", all_data$Cabin)] <- "E"
    # all_data$Deck[grep("F", all_data$Cabin)] <- "F"
    # all_data$Deck <- factor(all_data$Deck)

    ## Create normalised, numeric features 
    all_data$nAge <- normalise(all_data$Age)
    all_data$nFamilySize <- normalise(all_data$FamilySize)
    all_data$nFare <- normalise(all_data$Fare)
    all_data$nPclass <- normalise(all_data$Pclass)
    all_data$nSex <- as.numeric(all_data$Sex == 'female')
    all_data$nChild <- as.numeric(all_data$Sex == 'child')

    ## . . .

    ## Split the data back into a train set and a test set
    train <- data.frame(all_data[1:891,], train["Survived"])
    test <- all_data[892:1309,]

    return(list(train, test))
}

dataExploration <- function(train) {
    ## Record tables
    write.table(prop.table(table(train$Sex, train$Survived), margin=1),
                file="logs/tables.txt")

    write.table(prop.table(table(train$Child, train$Survived), margin=1),
                file="logs/tables.txt", append=TRUE)

    ## . . .
}

main <- function() {
    ## Prepare data
    data_set <- extractData()
    train <- data.frame(data_set[1])
    test <- data.frame(data_set[2])
    ## Analyse data
    dataExploration(train)

    ## Set fixed seed
    set.seed(111)
    ## Set cross validation folds
    k <- 10

    print("baseline")
    print(crossValidation(model_name="baseline", data=train, num_folds=k))

    print("decision tree")
    print(crossValidation(model_name="dTree", 
         params=rpart.control(minsplit=11, cp=0.001), data=train, num_folds=k))

    print("random forest")
    print(crossValidation(model_name="forest",
        c(200, 3), data=train, num_folds=k))

    print("k-nearest neighbours")
    print(crossValidation(model_name="knn", data=train, num_folds=k))

    print("support vector machines")
    print(crossValidation(model_name="svm", data=train, num_folds=k))

    ## Create best prediction

    # model_prediction <- modelPrediction("baseline", NULL, train, test)
    # solution <- data.frame(
    #     PassengerId=test$PassengerId, Survived=model_prediction)

    # model_prediction <- modelPrediction("dTree",
    #     params=rpart.control(cp=0.02, minsplit=20), train, test)
    # solution <- data.frame(
    #     PassengerId=test$PassengerId, Survived=model_prediction)

    model_prediction <- modelPrediction("forest", c(200, 3), train, test)
    solution <- data.frame(
        PassengerId=test$PassengerId, Survived=model_prediction)

    # model_prediction <- modelPrediction("knn", NULL, train, test)
    # solution <- data.frame(
    #      PassengerId=test$PassengerId, Survived=model_prediction)

    # model_prediction <- modelPrediction("svm", NULL, train, test)
    # solution <- data.frame(
    #     PassengerId=test$PassengerId, Survived=model_prediction)

    checkEquals(nrow(solution), nrow(test))
    write.csv(solution, file="predictions/forest.csv", row.names=FALSE)
}

main()
