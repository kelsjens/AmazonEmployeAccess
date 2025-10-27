## This is a PCA r script for my things
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
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  step_novel(all_nominal_predictors(), new_level = "new") %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = .75)


#Logistic Regression
logregmodel <- logistic_reg() %>% 
  set_engine('glm')

log_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(logregmodel) %>% 
  fit(data = train)

amazon_predictions <- predict(log_wf, 
                              new_data=test, 
                              type="prob") 

submission <- bind_cols(test 
                        %>% select(id),
                        amazon_predictions %>% 
                          transmute(ACTION = .pred_1)) %>%
  drop_na(ACTION)
vroom_write(x=submission, file = "./log_amazon_preds.csv", delim = ",")





# Penalized Regression
pen_model <- logistic_reg(mixture = tune(), penalty = tune()) %>% 
  set_engine('glmnet')

pen_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(pen_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 10)

folds <- vfold_cv(train, v = 10, repeats = 1)

cv_results <- pen_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

besttune <- cv_results %>% 
  select_best(metric = 'roc_auc')

final_wf <- pen_wf %>% 
  finalize_workflow(besttune) %>% 
  fit(data = train)

final_wf %>% 
  predict(new_data = test, type = "prob")

amazon_predictions <- predict(final_wf, 
                              new_data=test, 
                              type="prob") 
submission <- bind_cols(test 
                        %>% select(id),
                        amazon_predictions %>% 
                          transmute(ACTION = .pred_1)) %>%
  drop_na(ACTION)
vroom_write(x=submission, file = "./Penalized_amazon_preds.csv", delim = ",")



# Random Forest Model
train <- train %>% mutate(ACTION = factor(ACTION, levels = c(1, 0)))

# (Optional but recommended) stratified CV so class imbalance doesn’t break folds
folds <- vfold_cv(train, v = 5, repeats = 1, strata = ACTION)

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
  mtry(range = c(1, 3)),
  min_n(range = c(1, 3)),
  levels = 3
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









