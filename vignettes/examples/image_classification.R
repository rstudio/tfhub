#' In this example we will use pre-trained features from the Mobile Net model
#' to create an image classifier to the CIFAR-100 dataset.

library(keras)
library(tfhub)


# Get data ----------------------------------------------------------------

cifar <- dataset_cifar100()


# Build the model ---------------------------------------------------------

feature_model <- "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

input <- layer_input(shape = c(32, 32, 3))

resize_and_scale <- function(x) {
  tf$image$resize(x/255, size = shape(224, 224))
}

output <- input %>%
  layer_lambda(f = resize_and_scale) %>%
  layer_hub(handle = feature_model) %>%
  layer_dense(units = 10, activation = "softmax")

model <- keras_model(input, output)

model %>%
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

# Fitting -----------------------------------------------------------------

model %>%
  fit(
    x = cifar$train$x,
    y = cifar$train$y,
    validation_split = 0.2,
    batch_size = 128
  )

model %>%
  evaluate(x = cifar$test$x, y = cifar$test$y)


