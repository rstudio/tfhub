context("module")

source("utils.R")

test_succeeds("hub_module can load a module", {
  module <- hub_module("https://tfhub.dev/google/universal-sentence-encoder/1")
  expect_true(reticulate::py_has_attr(module, "get_signature_names"))
})

test_succeeds("hub_module_spec can load a module", {
  module_spec <- hub_load_module_spec("https://tfhub.dev/google/universal-sentence-encoder/1")
  module <- hub_module(module_spec)
  expect_true(reticulate::py_has_attr(module, "get_signature_names"))
})

