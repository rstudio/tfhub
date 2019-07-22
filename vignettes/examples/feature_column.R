#' In this example we will use the PetFinder dataset to demonstrate the
#' feature_spec functionality with TensorFlow Hub.

# TODO wait for https://github.com/tensorflow/hub/issues/333

library(keras)
library(tfhub)
library(tfdatasets)
library(readr)
library(dplyr)


# Read data ---------------------------------------------------------------

dataset <- read_csv("train.csv")


# Build the feature spec --------------------------------------------------

spec <- feature_spec(dataset, AdoptionSpeed ~ Description) %>%
  step_text_embedding_column(
    Description,
    module_spec = "https://tfhub.dev/google/elmo/2"
    )

spec <- fit(spec)

# Build the model ---------------------------------------------------------

inputs <- dataset %>%
  select(Description) %>%
  layer_input_from_dataset()

output <- inputs %>%
  layer_dense_features(spec$dense_features()) %>%
  layer_dense(units = 4, activation = "softmax")



