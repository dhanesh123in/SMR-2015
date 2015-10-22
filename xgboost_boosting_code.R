
#--------- L I B R A R Y ------------------------------------------------
library(xgboost)
library(readr)

#--------- P A R A M S ------------------------------------------------

param0 <- list(
  # general , non specific params - just guessing
  "objective"  = "binary:logistic"
  , "eval_metric" = "auc"
  , "eta" = 0.01
  , "subsample" = 0.7
  , "colsample_bytree" = 0.5
  , "min_child_weight" =6
  , "max_depth" = 9
  , "alpha" = 4
  , "nthreads" = 3
)

version="local"
subversion = 2
set.seed(1948 ^ subversion)

PROD = T

# -------- d a t a ---------------

path = "input/"
train <- read_csv(paste0(path, "train.csv", collapse = ""))
y <- train$target
train <- train[,-c(1, 1934)]
gc()
test <- read_csv(paste0(path, "test.csv", collapse = ""))
test <- test[,-1]
gc()
for (i in 1:ncol(train)) {
  if (class(train[, i]) == "character") {
    tmp= as.numeric(as.factor(c(train[,i], test[,i])))
    train[,i]<- head(tmp, nrow(train))
    test[,i]<- tail(tmp, nrow(test))
  }
}
train[is.na(train)] <- -9999
test[is.na(test)] <- -9999

# some simple feature cleaning/engineering boosts the LB-AUC by 0.0025. 

# -------- r u n s  i n 8 GB at home but not here ---------------

sumwpos = length(y[y==TRUE])
sumwneg = length(y[y==FALSE])

cnt=0



while (PROD){
  cnt=cnt+1
  cat("\n Run number..",cnt)
  start.time <- proc.time()[3]
  
  hold <- sample(1:nrow(train), nrow(train) * 0.1) #10% training data for stopping
  xgtrain = xgb.DMatrix(as.matrix(train[-hold,]), label = y[-hold], missing = NA)
  xgval = xgb.DMatrix(as.matrix(train[hold,]), label = y[hold], missing = NA)
  gc()
  watchlist <- list('val' = xgval)
  model = xgb.train(
    nrounds = 2500   # increase for more results at home
    , params = param0
    , data = xgtrain
    , early.stop.round = 50
    , watchlist = watchlist
    , print.every.n = 5
    , maximize = T
    , missing = -9999
  )
  cat("\nWatchlist best Score", model$bestScore)
  # Watchlist best Score 0.794478
  
  # -------- w r i t e   r e s u l t  ------
  
  if (model$bestScore > 0.80) {
    
    cat("\nProducing Guesstimate....")
    #rm("train")
    rm("xgval"); gc()
    train_val <-predict(model, newdata=xgtrain)
    rm("xgtrain")
    xgtest <- xgb.DMatrix(as.matrix(test), missing = NA)
    
    bst <- model$bestInd
    preds_out <- predict(model, xgtest, ntreelimit = bst)
    
    sub <-
      read_csv(paste0(path, "sample_submission.csv", collapse = ""))
    sub$target <- preds_out
    write_csv(sub, paste0(path, "test_submission_",subversion,"_",cnt,"_", model$bestScore ,".csv", collapse = ""))
    xgb.save(model,paste0(path, "test_submission_",subversion,"_",cnt,"_", model$bestScore ,".mod", collapse = ""))
    et <- -start.time + proc.time()[3]
    cat("\nTotal elapsed in seconds: ", et)
    cat("\nOutput okay: end ", format(Sys.time()),"\n")
  }
}
