#' This example is a demo of BigGAN image generators available on TF Hub.
#'
#' See [this jupyter notebook](https://github.com/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb) for more info.
#'
#' This example currently requires TensorFlow 2.0 Nightly preview.
#' It can be installed with
#' reticulate::py_install("tf-nightly-2.0-preview", pip = TRUE)
#'

# Setup -------------------------------------------------------------------

library(tensorflow)
library(tfhub)

module <- hub_load(handle = "https://tfhub.dev/deepmind/biggan-deep-256/1")

# ImageNet label ----------------------------------------------------------
# Select the ImageNet label you want to generate images for.

imagenet_labels <- jsonlite::fromJSON("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json")
label_id <- which(imagenet_labels == "tiger shark") - 1L

# Definitions -------------------------------------------------------------

# Sample random noise (z) and ImageNet label (y) inputs.
batch_size <- 8
truncation <- tf$constant(0.5)
z <- tf$random$truncated_normal(shape = shape(batch_size, 128)) * truncation

# create labels
y <- tf$one_hot(rep(label_id, batch_size), 1000L)

# Call BigGAN on a dict of the inputs to generate a batch of images with shape
# [8, 256, 256, 3] and range [-1, 1].
samples <- module$signatures[["default"]](y=y, z=z, truncation=truncation)

# Create plots ------------------------------------------------------------

create_plot <- function(samples, ncol) {

  images <- samples[[1]] %>%
    apply(1, function(x) {
      magick::image_read(as.raster((as.array(x) + 2)/4))
    }) %>%
    do.call(c, .)

  split(images, rep(1:ncol, lenght.out = length(images))) %>%
    lapply(magick::image_append, stack = TRUE) %>%
    do.call(c, .) %>%
    magick::image_append() %>%
    as.raster() %>%
    plot()
}

create_plot(samples, ncol = 4)


