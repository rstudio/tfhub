context("columns")

source("utils.R")

test_succeeds("hub_text_embedding_column can load a module", {

  column <- hub_text_embedding_column("sentence",
                                      "https://tfhub.dev/google/nnlm-en-dim128/1")
  expect_true(reticulate::py_has_attr(column, "name"))
})


