

library(tensorflow)
library(tfestimators)
library(tfdatasets)
library(tfhub)


# download imdb sentiment dataset if we need to
aclimdb_data_dir <- "aclImdb"
if (!dir.exists(aclimdb_data_dir)) {
  aclimdb_tar_file <- "aclImdb_v1.tar.gz"
  on.exit(unlink(aclimdb_tar_file), add = TRUE)
  download.file("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                destfile = aclimdb_tar_file)
  untar(aclimdb_tar_file)
}


# create an imdb dataset
imdb_dataset <- function(split, batch_size = 32) {

  polarity_dataset <- function(polarity, encoding) {

    # enumerate files for this split (train/test) & polarity (pos/neg)
    files <- file_list_dataset(file.path(aclimdb_data_dir, split, polarity, "*.txt"),
                               shuffle = TRUE)

    # map to record with review text
    files %>%
      dataset_map_and_batch(batch_size = batch_size, function(file) {
        list(
          text = tf$read_file(file),
          polarity = as.integer(encoding)
        )
      })
  }

  sample_from_datasets(list(
    polarity_dataset("pos", 1),
    polarity_dataset("neg", 0)
  ))
}

imdb_input_fn <- function(split) {
  imdb_dataset(split) %>%
    input_fn(features = "text", response = "polarity")
}


embedded_text_feature_column <- hub_text_embedding_column(
  key = "text",
  module_spec = "https://tfhub.dev/google/universal-sentence-encoder/1"
)

estimator <- dnn_classifier(
  hidden_units= c(500, 100),
  feature_columns = embedded_text_feature_column,
  n_classes = 2,
  optimizer = "Adagrad"
)


estimator %>% train(imdb_input_fn("train"))

estimator %>% evaluate(imdb_input_fn("test"))


# we should in theory be able to serve this with a REST API that takes
# string tensors (text) as input, and produces a predicted polarity
# as output

#estimator %>% export_savedmodel()
#library(tfdeploy)
#serve_savedmodel()








