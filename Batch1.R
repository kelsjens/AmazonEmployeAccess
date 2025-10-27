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


n_cores <- 15
cl <- makeCluster(n_cores)
doParallel::registerDoParallel(cl)


test  <- vroom("./amazon-employee-access-challenge/test.csv",  delim = ",")
train <- vroom("./amazon-employee-access-challenge/train.csv", delim = ",")



# # ── Dummy-encoding recipe (with <0.1% lumping). This is the one that matches your assignment. ──
# my_recipe <- recipe(ACTION ~ ., data = train) %>% 
#   step_mutate(across(everything(), as.factor)) %>%
#   step_other (all_nominal_predictors(), threshold = 0.001) %>%
#   step_dummy (all_nominal_predictors())
# 
# 
# 
# # prep & bake
# prep_obj    <- prep(my_recipe, training = train, verbose = TRUE)
# train_baked <- bake(prep_obj, new_data = train)
# ncol(train_baked)
# 
# 
# logregmodel <- logistic_reg() %>% 
#   set_engine('glm')
# 
# log_wf <- workflow() %>% 
#   add_recipe(my_recipe) %>% 
#   add_model(logregmodel) %>% 
#   fit(data = train)
# 
# amazon_predictions <- predict(log_wf, 
#                               new_data=test, 
#                               type="prob") 
# 
# submission <- bind_cols(test 
#                         %>% select(id),
#                         amazon_predictions %>% 
#                           transmute(ACTION = .pred_1)) %>%
#   drop_na(ACTION)
# vroom_write(x=submission, file = "./Logistic_amazon_preds.csv", delim = ",")
# 




# Penalized regression

# my_recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_mutate(across(everything(), as.factor)) %>%
#   step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
#   step_novel(all_nominal_predictors(), new_level = "new") %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# 
# pen_model <- logistic_reg(mixture = tune(), penalty = tune()) %>% 
#   set_engine('glmnet')
# 
# pen_wf <- workflow() %>% 
#   add_recipe(my_recipe) %>% 
#   add_model(pen_model)
# 
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 10)
# 
# folds <- vfold_cv(train, v = 10, repeats = 1)
# 
# cv_results <- pen_wf %>% 
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# besttune <- cv_results %>% 
#   select_best(metric = 'roc_auc')
# 
# final_wf <- pen_wf %>% 
#   finalize_workflow(besttune) %>% 
#   fit(data = train)
# 
# final_wf %>% 
#   predict(new_data = test, type = "prob")
# 
# amazon_predictions <- predict(final_wf, 
#                               new_data=test, 
#                               type="prob") 
# submission <- bind_cols(test 
#                         %>% select(id),
#                         amazon_predictions %>% 
#                           transmute(ACTION = .pred_1)) %>%
#   drop_na(ACTION)
# vroom_write(x=submission, file = "./Penalized_amazon_preds.csv", delim = ",")
# 
# 


# Random Forest Code
train <- train %>% mutate(ACTION = factor(ACTION, levels = c(1, 0)))

# (Optional but recommended) stratified CV so class imbalance doesn’t break folds
folds <- vfold_cv(train, v = 5, repeats = 1, strata = ACTION)

# 2) Recipe: do NOT turn everything into factors
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  step_novel(all_nominal_predictors(), new_level = "new") %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

# 3) Build a safe grid
# mtry must be <= # predictors after the recipe; keep your small demo grid but valid syntax
tuning_grid <- grid_regular(
  mtry(range = c(1, 9)),
  min_n(range = c(1, 9)),
  levels = 9
)

cv_results <- forest_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)   # uses level "1" as event
  )

besttune <- cv_results %>%
  select_best(metric = "roc_auc")

final_wf <- forest_wf %>%
  finalize_workflow(besttune) %>%
  fit(data = train)

amazon_predictions <- predict(final_wf, new_data = test, type = "prob")

submission <- bind_cols(
  test %>% select(id),
  amazon_predictions %>% transmute(ACTION = .pred_1)
) %>%
  drop_na(ACTION)

vroom_write(x = submission, file = "./Forest_amazon_preds.csv", delim = ",")










# # KNN Analysis
# my_recipe <- recipe(ACTION ~ ., data = train) %>%
#   step_mutate(across(everything(), as.factor)) %>%
#   step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
#   step_novel(all_nominal_predictors(), new_level = "new") %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# 
# 
# knn_model <- nearest_neighbor(neighbors = tune()) %>% 
#   set_mode("classification") %>% 
#   set_engine("kknn")
# 
# knn_wf <- workflow() %>% 
#   add_recipe(my_recipe) %>% 
#   add_model(knn_model)
# 
# # KNN tunes 'neighbors' (not mtry/min_n). Keep levels the same.
# tuning_grid <- grid_regular(
#   neighbors(range = c(1, 75)),
#   levels = 10
# )
# 
# folds <- vfold_cv(train, v = 10, repeats = 1)
# 
# cv_results <- knn_wf %>% 
#   tune_grid(
#     resamples = folds,
#     grid = tuning_grid,
#     metrics = metric_set(roc_auc)
#   )
# 
# besttune <- cv_results %>% 
#   select_best(metric = "roc_auc")
# 
# final_wf <- knn_wf %>% 
#   finalize_workflow(besttune) %>% 
#   fit(data = train)
# 
# # Use the finalized workflow to predict
# amazon_predictions <- final_wf %>% 
#   predict(new_data = test, type = "prob")
# 
# submission <- bind_cols(
#   test %>% select(id),
#   amazon_predictions %>% transmute(ACTION = .pred_1)
# ) %>% 
#   drop_na(ACTION)
# 
# vroom_write(x = submission, file = "./KNN_amazon_preds.csv", delim = ",")
# 
# 




# Naive Bayes
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate(across(everything(), as.factor)) %>%
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  step_novel(all_nominal_predictors(), new_level = "new") %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))


nb_model <- naive_Bayes(
  Laplace   = tune(),
  smoothness = tune()
) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")  # requires {naivebayes}

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Re samples
folds <- vfold_cv(train, v = 5)

## Tune smoothness and Laplace
# Build parameter set from the model then set reasonable ranges

tune_grid_nb <- grid_regular(Laplace(range = c(0,1)),
                             smoothness(.1,2),
                             levels = 10)

nb_res <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = tune_grid_nb,
    metrics = metric_set(roc_auc)  # use accuracy, pr_auc, etc. if preferred
  )

nb_best <- nb_res %>%
  select_best(metric = "roc_auc")

nb_final_wf <- nb_wf %>%
  finalize_workflow(nb_best) %>%
  fit(data = train)

## Predict
# Class probabilities
nb_probs <- predict(nb_final_wf, new_data = myNewData, type = "prob")


submission <- bind_cols(
  test %>% select(id),
  nb_probs %>% transmute(ACTION = .pred_1)
) %>% 
  drop_na(ACTION)

vroom_write(x = submission, file = "./bayes_amazon_preds.csv", delim = ",")
