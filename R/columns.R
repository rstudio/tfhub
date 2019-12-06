

#' Module to construct a dense representation from a text feature.
#'
#' This feature column can be used on an input feature whose values are strings of
#' arbitrary size.
#'
#' @inheritParams hub_sparse_text_embedding_column
#'
#' @export
hub_text_embedding_column <- function(key, module_spec, trainable = FALSE) {
  tfhub$text_embedding_column(
    key = key,
    module_spec = module_spec,
    trainable = trainable
  )
}

#' Module to construct dense representations from sparse text features.
#'
#' The input to this feature column is a batch of multiple strings with
#' arbitrary size, assuming the input is a SparseTensor.
#'
#' This type of feature column is typically suited for modules that operate
#' on pre-tokenized text to produce token level embeddings which are combined
#' with the combiner into a text embedding. The combiner always treats the tokens
#'  as a bag of words rather than a sequence.
#'
#' The output (i.e., transformed input layer) is a DenseTensor, with
#' shape [batch_size, num_embedding_dim].
#'
#' @param key A string or [feature_column](https://tensorflow.rstudio.com/tfestimators/articles/feature_columns.html)
#'  identifying the text feature.
#' @param module_spec A string handle or a _ModuleSpec identifying the module.
#' @param combiner a string specifying reducing op for embeddings in the same Example.
#'  Currently, 'mean', 'sqrtn', 'sum' are supported. Using `combiner = NULL` is
#'  undefined.
#' @param default_value default value for Examples where the text feature is empty.
#'  Note, it's recommended to have default_value consistent OOV tokens, in case
#'  there was special handling of OOV in the text module. If `NULL`, the text
#'  feature is assumed be non-empty for each Example.
#' @param trainable Whether or not the Module is trainable. `FALSE` by default,
#'  meaning the pre-trained weights are frozen. This is different from the ordinary
#'  `tf.feature_column.embedding_column()`, but that one is intended for training
#'  from scratch.
#'
#' @export
hub_sparse_text_embedding_column <- function(key, module_spec, combiner,
                                             default_value, trainable = FALSE) {
  tfhub$sparse_text_embedding_column(
    key = key,
    module_spec = module_spec,
    combiner = combiner,
    default_value = default_value,
    trainable = trainable
  )
}


#' Module to construct a dense 1-D representation from the pixels of images.
#'
#' @inheritParams hub_sparse_text_embedding_column
#'
#' @details
#' This feature column can be used on images, represented as float32 tensors of RGB pixel
#' data in the range [0,1].
#'
#' @export
hub_image_embedding_column <- function(key, module_spec) {
  tfhub$image_embedding_column(
    key = key,
    module_spec = module_spec
  )
}


