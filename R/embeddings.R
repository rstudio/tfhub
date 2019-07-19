# Exporter tool for TF-Hub text embedding modules

load_file <- function(file_path) {
  suppressMessages({
    embeddings <- vroom::vroom(file_path, delim = " ", quote = "", col_names = FALSE)
  })

  list(
    vocabulary = embeddings$X1,
    embeddings = as.matrix(embeddings[, -1])
  )
}

tokenize <- tf$`function`(reticulate::py_func(function(sentences, lookup_table) {
  # Perform a minimalistic text preprocessing by removing punctuation and
  # splitting on spaces.
  normalized_sentences <- tf$strings$regex_replace(
    input=sentences, pattern = "\\pP", rewrite="")
  normalized_sentences <- tf$reshape(normalized_sentences, list(-1))
  sparse_tokens <- tf$strings$split(normalized_sentences, " ")$to_sparse()

  # Deal with a corner case: there is one empty sentence.
  sparse_tokens <- tf$sparse$fill_empty_rows(sparse_tokens, tf.constant(""))[[1]]
  # Deal with a corner case: all sentences are empty.
  sparse_tokens <- tf$sparse$reset_shape(sparse_tokens)
  sparse_token_ids <- lookup_table.lookup(sparse_tokens$values)
}), input_signature = list(tf$TensorSpec(shape = tensorflow::shape(NULL), dtype = tf$string)))

export_embedding_file <- function(file_path) {

  data <- load_file(file_path)
  lookup_table <- tf$lookup$KeyValueTensorInitializer(keys = data$vocabulary, )



}
