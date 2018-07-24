



#' Re-usable TensorFlow module
#'
#' A Module represents a part of a TensorFlow graph that can be exported to disk (based on
#' the SavedModel format) and later re-loaded. A Module has a defined interface that
#' allows it to be used in a replaceable way, with little or no knowledge of its internals
#' and its serialization format
#'
#' @param module_spec Module spec loaded using [hub_load_module_spec()] or string
#'   describing the location of a module. There are several supported path encoding
#'   schemes: a) URL location specifying an archived module (e.g.
#'   http://domain/module.tgz) b) Any filesystem location of a module directory (e.g.
#'   /module_dir for a local filesystem). All filesystems implementations provided by
#'   Tensorflow are supported.
#' @param trainable Whether the Module is trainable. If `FALSE`, no variables are added to
#'   TRAINABLE_VARIABLES collection, and no tensors are added to REGULARIZATION_LOSSES
#'   collection.
#' @param name A string, the variable scope name under which to create the Module. It will
#'   be uniquified and the equivalent name scope must be unused.
#' @param tags A set of strings specifying the graph variant to use.
#'
#' @export
hub_module <- function(module_spec, trainable = FALSE, name = "module", tags = NULL) {
  tfhub$Module(
    module_spec,
    trainable = trainable,
    name = name,
    tags = tags
  )
}

#' Loads a module specification from the filesystem or a URL
#'
#' @param path String describing the location of a module. There are several supported
#'   path encoding schemes: a) URL location specifying an archived module (e.g.
#'   http://domain/module.tgz) b) Any filesystem location of a module directory (e.g.
#'   /module_dir for a local filesystem). All filesystems implementations provided by
#'   Tensorflow are supported.
#'
#' @export
hub_load_module_spec <- function(path) {
  tfhub$load_module_spec(path.expand(path))
}





