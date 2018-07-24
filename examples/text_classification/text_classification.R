

library(keras)
library(tfdatasets)

# download imdb sentiment dataset if we need to
aclimdb_data_dir <- "aclimdb"
if (!dir.exists(aclimdb_data_dir)) {
  aclimdb_tar_file <- "aclImdb_v1.tar.gz"
  on.exit(unlink(aclimdb_tar_file), add = TRUE)
  download.file("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                destfile = aclimdb_tar_file)
  untar(aclimdb_tar_file)
}


# create an imdb dataset
imdb_dataset <- function(split) {

  polarity_dataset <- function(polarity) {

    # enumerate files for this split (train/test) & polarity (pos/neg)
    files <- file_list_dataset(file.path(aclimdb_data_dir, split, polarity$name, "*.txt"),
                               shuffle = TRUE)

    # map to record with review text and integer encoding of polarity
    files %>%
      dataset_map(function(file) {
        list(
          text = tf$read_file(file),
          polarity = polarity$encoding
        )
      })
  }

  # return dataset with positive and negative reviews interleaved
  polarities <- data.frame(name = c("pos", "neg"), encoding = c(1L,0L))
  tensor_slices_dataset(polarities) %>%
    dataset_interleave(polarity_dataset, cycle_length = 2)
}

# create training and test datasets
train_dataset <- imdb_dataset("train")
test_dataset <- imdb_dataset("test")





