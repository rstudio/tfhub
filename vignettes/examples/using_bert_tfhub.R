#' In this example we use tfhub to obtain pre-trained word-embeddings from the
#' BERT model.
#'
#' This example is based on [this colab notebook](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#scrollTo=hhbGEfwgdEtw)
#'
#' The dataset comes from the Toxic Comment Classification Challenge in Kaggle and
#' can be downlaoded here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
#'

library(keras)
library(tfhub)
library(readr)
library(pins)


# Read data ---------------------------------------------------------------

comments <- read_csv("train.csv.zip")

ind_train <- sample.int(nrow(comments), 0.8*nrow(comments))
train <- comments[ind_train,]
test <- comments[-ind_train,]

# Build the model ---------------------------------------------------------

# We the token based text embedding trained on English Google News 130GB corpus.
# https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1
# The model is available at the above URL.

embeddings <- layer_hub(
  handle = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
  trainable = FALSE
)

input <- layer_input(shape = shape(), dtype = "string")

output <- input %>%
  embeddings() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 6, activation = "sigmoid")

model <- keras_model(input, output)

model %>%
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

# Fit the model -----------------------------------------------------------

model %>%
  fit(
    x = train$comment_text,
    y = as.matrix(train[,-c(1:2)]),
    validation_split = 0.2
  )

model %>%
  evaluate(x = test$comment_text, y = as.matrix(test[,-c(1,2)]))

# Calculating the AUC for each class
purrr::map2_dbl(
  as.data.frame(actual),
  as.data.frame(preds),
  Metrics::auc
)
