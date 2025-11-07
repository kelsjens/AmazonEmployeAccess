library(tidyverse)
library(tidymodels)
library(vroom)
library(dplyr)
library(tune)
library(ggplot2)
library(embed)
library(doParallel)
library(kknn)
library(discrim)
library(remotes)

# read in data
test  <- vroom("./amazon-employee-access-challenge/test.csv",  delim = ",")
train <- vroom("./amazon-employee-access-challenge/train.csv", delim = ",")

# The recipe
train <- train %>% mutate(ACTION = factor(ACTION, levels = c(1, 0)))


base_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate(across(everything(), as.factor)) %>%
  step_other (all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy (all_nominal_predictors()) %>% 
  step_normalize(all_predictors())%>% 
  step_pca(all_predictors(), threshold = .15)

prep_obj <- prep(base_recipe, training = train, verbose = TRUE)
train_baked <- bake(prep_obj, new_data = train)
ncol(train_baked)





# ---------- Resamples ----------
folds <- vfold_cv(train, v = 5, strata = ACTION)

# Optional tuning control
ctrl <- control_grid(save_pred = TRUE)

# =======================================================
# 1) Polynomial SVM
svmPoly <- svm_poly(degree = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svmPoly_wf <- workflow() %>%
  add_recipe(base_recipe) %>%
  add_model(svmPoly)

grid_poly <- grid_regular(
  degree(range = c(1, 3)),   # <- remove the L suffix
  cost(),
  levels = 5
)

svmPoly_res <- svmPoly_wf %>%
  tune_grid(resamples = folds, grid = grid_poly, metrics = metric_set(roc_auc, accuracy), control = ctrl)

svmPoly_best <- svmPoly_res %>% select_best(metric = "roc_auc")
svmPoly_fit  <- finalize_workflow(svmPoly_wf, svmPoly_best) %>% fit(train)

pred_poly <- predict(svmPoly_fit, new_data = test, type = "prob")
sub_poly <- bind_cols(
  test %>% select(RESOURCE),                # change if your ID column is different
  pred_poly %>% transmute(ACTION = .pred_1)
) %>% drop_na(ACTION)

vroom_write(sub_poly, file = "./svm_poly_amazon_preds.csv", delim = ",")

# =======================================================
# 2) RBF (Radial) SVM
svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svmRadial_wf <- workflow() %>%
  add_recipe(base_recipe) %>%
  add_model(svmRadial)

grid_rbf <- grid_regular(
  rbf_sigma(),                 # log10 scale by default
  cost(),
  levels = 5
)

svmRadial_res <- svmRadial_wf %>%
  tune_grid(resamples = folds, grid = grid_rbf, metrics = metric_set(roc_auc, accuracy), control = ctrl)

svmRadial_best <- svmRadial_res %>% select_best(metric = "roc_auc")
svmRadial_fit  <- finalize_workflow(svmRadial_wf, svmRadial_best) %>% fit(train)

pred_rbf <- predict(svmRadial_fit, new_data = test, type = "prob")
sub_rbf <- bind_cols(
  test %>% select(RESOURCE),
  pred_rbf %>% transmute(ACTION = .pred_1)
) %>% drop_na(ACTION)

vroom_write(sub_rbf, file = "./svm_rbf_amazon_preds.csv", delim = ",")

# =======================================================
# 3) Linear SVM
svmLinear <- svm_linear(cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svmLinear_wf <- workflow() %>%
  add_recipe(base_recipe) %>%
  add_model(svmLinear)

grid_lin <- grid_regular(cost(), levels = 10)

svmLinear_res <- svmLinear_wf %>%
  tune_grid(resamples = folds, grid = grid_lin, metrics = metric_set(roc_auc, accuracy), control = ctrl)

svmLinear_best <- svmLinear_res %>% select_best(metric = "roc_auc")
svmLinear_fit  <- finalize_workflow(svmLinear_wf, svmLinear_best) %>% fit(train)

pred_lin <- predict(svmLinear_fit, new_data = test, type = "prob")
sub_lin <- bind_cols(
  test %>% select(RESOURCE),
  pred_lin %>% transmute(ACTION = .pred_1)
) %>% drop_na(ACTION)

vroom_write(sub_lin, file = "./svm_linear_amazon_preds.csv", delim = ",")