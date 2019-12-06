# tfhub

<!-- badges: start -->
[![Actions Status](https://github.com/rstudio/tfhub/workflows/R-CMD-check/badge.svg)](https://github.com/rstudio/tfhub/)
<!-- badges: end -->

The tfhub package provides R wrappers to [TensorFlow Hub](https://www.tensorflow.org/hub).

[TensorFlow Hub](https://www.tensorflow.org/hub) is a library for reusable machine learning modules.

TensorFlow Hub is a library for the publication, discovery, and consumption of reusable parts of machine learning models. A module is a self-contained piece of a TensorFlow graph, along with its weights and assets, that can be reused across different tasks in a process known as transfer learning. Transfer learning can:

* Train a model with a smaller dataset,
* Improve generalization, and
* Speed up training.

## Installation

You can install the development version from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("rstudio/tfhub")
```

After installing the tfhub package you need to install the TensorFlow Hub python 
module:

``` r
library(tfhub)
install_tfhub()
```

Go to [**the website**](https://tensorflow.rstudio.com/guide/tfhub/intro/) for more information.
