context("load")

test_succeeds("Can load module from URL", {
  module <- hub_load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
  expect_s3_class(module, "tensorflow.python.saved_model.load._UserObject")
})

test_succeeds("Can load module from file path", {

  skip("Currently skipping due to bug exporting models on Windows")

  library(keras)

  input <- layer_input(shape = shape(1))
  input2 <- layer_input(shape = shape(1))
  output <- layer_add(list(input, input2))

  model <- keras_model(list(input, input2), output)

  tmp <- tempfile()
  dir.create(tmp)

  export_savedmodel(model, tmp, remove_learning_phase = FALSE)

  module <- hub_load(tmp)
  expect_s3_class(module, "tensorflow.python.saved_model.load._UserObject")

  expect_equal(
    as.numeric(module(list(tf$ones(shape = c(1,1)), tf$ones(shape = c(1,1))))),
    2
  )
})

test_succeeds("hub_load correctly uses the env var", {

  tmp <- tempfile()

  x <- callr::r(
    function() {
      tfhub::hub_load('https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4')
    },
    env = c(TFHUB_CACHE_DIR = tmp)
  )

  expect_length(list.files(tmp), 2)
})


