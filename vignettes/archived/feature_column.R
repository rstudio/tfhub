#' In this example we will use the PetFinder dataset to demonstrate the
#' feature_spec functionality with TensorFlow Hub.
#'
#' Currently, we need TensorFlow 2.0 nightly and disable eager execution
#' in order for this example to work.
#'
#' Waiting for https://github.com/tensorflow/hub/issues/333
#'
#'
#'

# Notes about why this was archived: https://github.com/rstudio/tfdatasets/issues/81
#
# Snippets to download the data if we want to restore this example:
#
# pip install kaggle
# login to kaggle.com, download API 'kaggle.json' file
# mkdir ~/.kaggle
# mv ~/kaggle.json ~/.kaggle/
# chmod 600 ~/.kaggle/kaggle.json
# kaggle competitions download -c petfinder-adoption-prediction

# unzip("petfinder-adoption-prediction.zip", exdir = "petfinder")


library(keras)
library(tfhub)
library(tfdatasets)
library(readr)
library(dplyr)

tf$compat$v1$disable_eager_execution()

# Read data ---------------------------------------------------------------

dataset <- read_csv("petfinder/train/train.csv") %>%
  filter(PhotoAmt > 0) %>%
  mutate(img_path = path.expand(paste0("petfinder/train_images/", PetID, "-1.jpg"))) %>%
  mutate_at(vars(Breed1:Health, State), as.character) %>%
  sample_n(size = nrow(.)) # shuffle

dataset_tf <- dataset %>%
  tensor_slices_dataset() %>%
  dataset_map(function(x) {
    img <- tf$io$read_file(filename = x$img_path) %>%
      tf$image$decode_jpeg(channels = 3L) %>%
      tf$image$resize(size = c(224L, 224L))
    x[["img"]] <- img/255
    x
  })

dataset_test <- dataset_tf %>%
  dataset_take(nrow(dataset)*0.2) %>%
  dataset_batch(512)

dataset_train <- dataset_tf %>%
  dataset_skip(nrow(dataset)*0.2) %>%
  dataset_batch(32)

# Build the feature spec --------------------------------------------------

spec <- dataset_train %>%
  feature_spec(AdoptionSpeed ~ .) %>%
  step_text_embedding_column(
    Description,
    module_spec = "https://tfhub.dev/google/universal-sentence-encoder/2"
    ) %>%
  step_image_embedding_column(
    img,
    module_spec = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3"
  ) %>%
  # step_pretrained_text_embedding(
  #   Description,
  #   handle = "https://tfhub.dev/google/universal-sentence-encoder/2"
  #   ) %>%
  # step_pretrained_text_embedding(
  #   img,
  #   handle = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3"
  # ) %>%
  step_numeric_column(Age, Fee, Quantity, normalizer_fn = scaler_standard()) %>%
  step_categorical_column_with_vocabulary_list(
    has_type("string"), -Description, -RescuerID, -img_path, -PetID, -Name
  ) %>%
  step_embedding_column(Breed1:Health, State)

spec <- fit(spec)

# Build the model ---------------------------------------------------------

inputs <- layer_input_from_dataset(dataset_train) %>% reticulate::py_to_r()
inputs <- inputs[-which(names(inputs) == "AdoptionSpeed")]

output <- inputs %>%
  layer_dense_features(spec$dense_features()) %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")

model <- keras_model(inputs, output)

model %>%
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

# Fit the model -----------------------------------------------------------

sess <- k_get_session()
sess$run(tf$compat$v1$initialize_all_variables())
sess$run(tf$compat$v1$initialize_all_tables())

model %>%
  fit(
    x = dataset_use_spec(dataset_train, spec),
    validation_data = dataset_use_spec(dataset_test, spec),
    epochs = 5
  )


