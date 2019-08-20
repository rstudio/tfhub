#' In this example we use tfhub and recipes to obtain pre-trained sentence embeddings.
#' We then firt a logistic regression model.
#'
#' The dataset comes from the Toxic Comment Classification Challenge in Kaggle and
#' can be downlaoded here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
#'

library(tfhub)
library(readr)
library(tidymodels)

# Read data ---------------------------------------------------------------

comments <- read_csv("train.csv.zip")

ind_train <- sample.int(nrow(comments), 0.8*nrow(comments))
train <- comments[ind_train,]
test <- comments[-ind_train,]

# Create our recipe specification -----------------------------------------

rec <- recipe(
  obscene ~ comment_text,
  data = train
  ) %>% step_pretrained_text_embedding(
    comment_text,
    handle = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
  ) %>%
  step_bin2factor(obscene)

rec <- prep(rec)

# Train glmnet ------------------------------------------------------------

logistic_fit <-
  logistic_reg() %>%
  set_mode("classification") %>%
  set_engine("glm") %>%
  fit(obscene ~ ., data = juice(rec))

logistic_fit$fit

# Results -----------------------------------------------------------------

test_embedded <- bake(rec, test)

test_results <- test_embedded %>%
  select(obscene) %>%
  mutate(
    class = predict(logistic_fit, new_data = test_embedded) %>%
      pull(.pred_class),
    prob  = predict(logistic_fit, new_data = test_embedded, type = "prob") %>%
      pull(.pred_yes)
  )

test_results %>% roc_auc(truth = obscene, prob)
test_results %>% accuracy(truth = obscene, class)
test_results %>% conf_mat(truth = obscene, class)

