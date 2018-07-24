

#' Module to construct a dense representation from a text feature.
#'
#' This feature column can be used on an input feature whose values are strings of
#' arbitrary size.
#'
#' @inheritParams hub_module
#'
#' @param key A string or
#'   \href{feature_column}{https://tensorflow.rstudio.com/tfestimators/articles/feature_columns.html}
#'    identifying the text feature.
#'
#' @export
hub_text_embedding_column <- function(key, module_spec, trainable = FALSE) {
  tfhub$text_embedding_column(
    key = key,
    module_spec = module_spec,
    trainable = trainable
  )
}


#' Module to construct a dense 1-D representation from the pixels of images.
#'
#' @inheritParams hub_module
#'
#' @param key A string or
#'   \href{feature_column}{https://tensorflow.rstudio.com/tfestimators/articles/feature_columns.html}
#'    identifying the input image data.
#'
#' @details
#' This feature column can be used on images, represented as float32 tensors of RGB pixel
#' data in the range [0,1]. This can be read from a
#' \href{column_numeric()}{https://tensorflow.rstudio.com/tfestimators/reference/column_numeric.html}
#' if the input data happens to have decoded images, all with the same shape [height,
#' width, 3]. More commonly, the
#' \href{input_fn}{https://tensorflow.rstudio.com/tfestimators/articles/input_functions.html}
#' will have code to explicitly decode images, resize them (possibly after performing data
#' augmentation such as random crops etc.), and provide a batch of shape [batch_size,
#' height, width, 3].
#'
#' @export
hub_image_embedding_column <- function(key, module_spec) {
  tfhub$image_embedding_column(
    key = key,
    module_spec = module_spec
  )
}
