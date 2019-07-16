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

#' Adds a signature to the module definition.
#'
#' @param name Signature name as a string. If omitted, it is interpreted as
#'   'default' and is the signature used when `Module.__call__` signature is
#'   not specified.
#' @param inputs A list from input name to Tensor or SparseTensor to feed when
#'   applying the signature. If a single tensor is passed, it is interpreted as a
#'   list with a single 'default' entry.
#' @param outputs A list from output name to Tensor or SparseTensor to return from
#'   applying the signature. If a single tensor is passed, it is interpreted as a
#'   list with a single 'default' entry.
#'
#' @note This must be called within a module_fn that is defining a Module.
#'
#' @export
hub_add_signature <- function(name = NULL, inputs = NULL, outputs = NULL) {
  tfhub$add_signature(name, inputs, outputs)
}

#' Creates a ModuleSpec from a function that builds the module's graph.
#'
#' The module_fn is called on a new graph (not the current one) to build the graph
#' of the module and define its signatures via `hub_add_signature()`.
#'
#' @note In anticipation of future TF-versions, `module_fn` is called on a graph
#'   that uses resource variables by default. If you want old-style variables then
#'   you can use with `tf$variable_scope("", use_resource=False)` in `module_fn`.
#'
#' @param module_fn a function to build a graph for the Module.
#' @param saved_model_path Directory with the SavedModel to use.
#' @param tags_and_args Optional list of tuples (tags, kwargs) of tags and keyword
#'   args used to define graph variants. If omitted, it is interpreted as `[(set(), {})]`,
#'   meaning module_fn is called once with no args.
#' @param drop_collections: list of collection to drop.
#'
#' @seealso hub_add_signature
#'
#' @rdname hub_create_module_spec
#' @export
hub_create_module_spec <- function(module_fn, tags_and_args = NULL, drop_collections = NULL) {
  tfhub$create_module_spec(module_fn, tags_and_args, drop_collections)
}

#' @rdname hub_create_module_spec
#' @export
hub_create_module_spec_from_savedmodel <- function(saved_model_path, drop_collections = NULL) {
  tfhub$create_module_spec_from_saved_model(
    saved_model_path = path.expand(saved_model_path),
    drop_collections = drop_collections
  )
}





