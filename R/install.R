
#' Install TensorFlow Hub
#'
#' This function is used to install the TensorFlow Hub python module.
#'
#' @param version version of TensorFlow Hub to be installed.
#' @param ... other arguments passed to [reticulate::py_install()].
#' @param restart_session Restart R session after installing (note this will
#'  only occur within RStudio).
#'
#' @export
install_tfhub <- function(version = "release", ..., restart_session = TRUE) {

  if (version == "nightly")
    module_string <- "tf-hub-nightly"
  else if (is.null(version) || version %in% c("release", "default", ""))
    module_string <- "tensorflow_hub"
  else
    module_string <- paste0("tensorflow_hub==", version)

  reticulate::py_install(packages = module_string, pip = TRUE, ...)

  if (restart_session && rstudioapi::hasFun("restartSession"))
    rstudioapi::restartSession()
}
