

skip_if_no_tfhub <- function(required_version = NULL) {
  if (!reticulate::py_module_available("tensorflow_hub"))
    skip("TensorFlow Hub not available for testing")
}

test_succeeds <- function(desc, expr) {
  test_that(desc, {
    skip_if_no_tfhub()
    expect_error(force(expr), NA)
  })
}
