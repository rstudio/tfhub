

# main tfhub module
tfhub <- NULL

.onLoad <- function(libname, pkgname) {
  tfhub <<- reticulate::import("tensorflow_hub", delay_load = list(
    priority = 10,
    environment = "r-tensorflow"
  ))
}
