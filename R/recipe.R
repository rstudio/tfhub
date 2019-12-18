#' Pretrained text-embeddings
#'
#' `step_pretrained_text_embedding` creates a *specification* of a
#'  recipe step that will transform text data into its numerical
#'  transformation based on a pretrained model.
#'
#' @param recipe A recipe object. The step will be added to the
#'  sequence of operations for this recipe.
#' @param ... One or more selector functions to choose variables.
#' @param role Role for the created variables
#' @param trained A logical to indicate if the quantities for
#'  preprocessing have been estimated.
#' @param skip A logical. Should the step be skipped when the
#'  recipe is baked by [recipes::bake.recipe()]? While all operations are baked
#'  when [recipes::prep.recipe()] is run, some operations may not be able to be
#'  conducted on new data (e.g. processing the outcome variable(s)).
#'  Care should be taken when using `skip = TRUE` as it may affect
#'  the computations for subsequent operations
#' @param handle the Module handle to resolve.
#' @param args other arguments passed to [hub_load()].
#' @param id A character string that is unique to this step to identify it.
#'
#' @examples
#'
#' \dontrun{
#' library(tibble)
#' library(recipes)
#' df <- tibble(text = c('hi', "heello", "goodbye"), y = 0)
#'
#' rec <- recipe(y ~ text, df)
#' rec <- rec %>% step_pretrained_text_embedding(
#'  text,
#'  handle = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
#' )
#'
#' }
#'
#' @export
step_pretrained_text_embedding <- function(
  recipe, ...,
  role = "predictor",
  trained = FALSE,
  handle,
  args = NULL,
  skip = FALSE,
  id = recipes::rand_id("pretrained_text_embedding")
) {

  terms <- recipes::ellipse_check(...)

  recipes::add_step(
    recipe,
    step_pretrained_text_embedding_new(
      terms = terms,
      trained = trained,
      role = role,
      vars = NULL,
      handle = handle,
      args = args,
      skip = skip,
      id = id
    )
  )
}

step_pretrained_text_embedding_new <- function(terms, role, trained, vars,
                                               handle, args, skip, id) {
    recipes::step(
      subclass = "pretrained_text_embedding",
      terms = terms,
      role = role,
      trained = trained,
      vars = vars,
      handle = handle,
      args = args,
      skip = skip,
      id = id
    )
}

#' Prep method for step_pretrained_text_embedding
#'
#' @param x object
#' @param info variables state
#' @param training wether or not it's training
#'
#' @inheritParams step_pretrained_text_embedding
#'
#' @export
prep.step_pretrained_text_embedding <- function(x, training, info = NULL, ...) {
  col_names <- recipes::terms_select(terms = x$terms, info = info)

  step_pretrained_text_embedding_new(
    terms = x$terms,
    trained = TRUE,
    role = x$role,
    vars = col_names,
    handle = x$handle,
    args = x$args,
    skip = x$skip,
    id = x$id
  )
}

get_embedding <- function(column, module) {
  out <- module(as.character(column))

  if (!tensorflow::tf$executing_eagerly()) {
    sess <- tensorflow::tf$compat$v1$Session()
    sess$run(tensorflow::tf$compat$v1$global_variables_initializer())
    sess$run(tensorflow::tf$compat$v1$tables_initializer())
    out <- sess$run(out)
    sess$close()
  } else {
    out <- as.matrix(out)
  }

  out
}

#' Bake method for step_pretrained_text_embedding
#'
#' @param object object
#' @param new_data new data to apply transformations
#'
#' @inheritParams step_pretrained_text_embedding
#'
#' @export
bake.step_pretrained_text_embedding <- function(object, new_data, ...) {

  module <- do.call(hub_load, append(list(handle = object$handle), object$args))

  embeddings <- lapply(object$vars, function(x) {
    embedding <- get_embedding(new_data[[x]], module)
    colnames(embedding) <- sprintf("%s_txt_emb_%04d", x, 1:ncol(embedding))
    tibble::as_tibble(embedding)
  })

  out <- do.call(cbind, append(list(new_data), embeddings))

  # remove text columns
  for (i in object$vars) {
    out[[i]] <- NULL
  }

  out
}


