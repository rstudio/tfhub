source("utils.R")

test_succeeds("layer_hub works with sequential models", {

  library(keras)

  model <- keras_model_sequential() %>%
    layer_hub(
      handle = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
      input_shape = c(224, 224, 3)
      ) %>%
    layer_dense(1)

  a <- tf$constant(array(0, dim = as.integer(c(1, 224, 224, 3))), dtype = "float32")

  res <- as.numeric(model(a))

  expect_is(res, "numeric")
})

test_succeeds("layer_hub works with functional API", {

  input <- layer_input(shape = c(224, 224, 3))

  output <- input %>%
    layer_hub(
      handle = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    ) %>%
    layer_dense(1)

  model <- keras_model(input, output)

  a <- tf$constant(array(0, dim = c(1, 224, 224, 3)), dtype = "float32")

  res <- as.numeric(model(a))

  expect_is(res, "numeric")
})

test_succeeds("can initialiaze the layer_hub", {

  features <- layer_hub(
    handle = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
  )

  input <- layer_input(shape = c(224, 224, 3))

  output <- input %>%
    features() %>%
    layer_dense(1)

  model <- keras_model(input, output)

  a <- tf$constant(array(0, dim = c(1, 224, 224, 3)), dtype = "float32")

  res <- as.numeric(model(a))

  expect_is(res, "numeric")
})


