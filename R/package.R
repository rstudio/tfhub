

# main tfhub module
tfhub <- NULL

.onLoad <- function(libname, pkgname) {
  tfhub <<- reticulate::import("tensorflow_hub", delay_load = list(
    priority = 10,
    environment = "r-tensorflow"
  ))

  vctrs::s3_register("recipes::prep", "step_pretrained_text_embedding")
  vctrs::s3_register("recipes::bake", "step_pretrained_text_embedding")
}
