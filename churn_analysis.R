##############################################################
# HarvardX PH125.9x - Data Science: Capstone
# Choose Your Own Project
# Customer Churn Prediction
##############################################################
#
# This script downloads the dataset, cleans it, explores it,
# and then trains two different models (logistic regression
# and random forest) to predict whether a customer will
# churn or not.
#
# NOTE: the dataset is hosted on my github repo so it can be
# downloaded automatically - no manual download needed.
##############################################################

# ------------------------------------------------------------
# 0. Install / load packages
# ------------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(corrplot)
library(scales)

# ------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------
# I am hosting a copy of the dataset on github so that the
# script can grab it automatically. If for some reason the
# download fails, the csv is also included in the submission
# folder as a backup.

url <- "https://raw.githubusercontent.com/USERNAME/customer-churn-capstone/main/customer_churn_business_dataset.csv"

dl <- tempfile()
download_ok <- tryCatch({
  download.file(url, dl)
  TRUE
}, error = function(e) FALSE)

if(download_ok){
  churn <- read_csv(dl)
} else {
  # fallback - local copy
  churn <- read_csv("customer_churn_business_dataset.csv")
}

dim(churn)
glimpse(churn)

# ------------------------------------------------------------
# 2. Data cleaning
# ------------------------------------------------------------

# checking for missing values
colSums(is.na(churn))

# the only column with NAs is complaint_type - this makes sense,
# a customer who never filed a complaint just has NA here.
# I'll recode that as "None" rather than dropping rows, since
# dropping ~2000 rows out of 10000 would throw away a lot of info.

churn <- churn %>%
  mutate(complaint_type = ifelse(is.na(complaint_type), "None", complaint_type))

# convert the target and the categorical predictors to factors
churn <- churn %>%
  mutate(
    churn = factor(churn, levels = c(0,1), labels = c("No","Yes")),
    gender = factor(gender),
    country = factor(country),
    city = factor(city),
    customer_segment = factor(customer_segment),
    signup_channel = factor(signup_channel),
    contract_type = factor(contract_type),
    payment_method = factor(payment_method),
    discount_applied = factor(discount_applied),
    price_increase_last_3m = factor(price_increase_last_3m),
    complaint_type = factor(complaint_type),
    survey_response = factor(survey_response, levels = c("Unsatisfied","Neutral","Satisfied"))
  )

# drop customer_id - it's just an identifier, no predictive value
churn <- churn %>% select(-customer_id)

# quick sanity check
summary(churn$churn)
mean(churn$churn == "Yes")

# ------------------------------------------------------------
# 3. Exploratory data analysis
# ------------------------------------------------------------

# 3.1 churn rate overall
churn %>%
  count(churn) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(churn, pct, fill = churn)) +
  geom_col() +
  geom_text(aes(label = percent(pct)), vjust = -0.3) +
  scale_y_continuous(labels = percent) +
  labs(title = "Overall churn rate", x = "Churn", y = "Percentage of customers") +
  theme_minimal()

# about 10% of customers in this dataset churned - so we have
# a pretty imbalanced target. Worth keeping in mind for modeling.

# 3.2 churn by contract type
churn %>%
  group_by(contract_type) %>%
  summarise(churn_rate = mean(churn == "Yes")) %>%
  ggplot(aes(contract_type, churn_rate, fill = contract_type)) +
  geom_col() +
  scale_y_continuous(labels = percent) +
  labs(title = "Churn rate by contract type", x = "Contract type", y = "Churn rate") +
  theme_minimal()

# 3.3 churn by customer segment
churn %>%
  group_by(customer_segment) %>%
  summarise(churn_rate = mean(churn == "Yes")) %>%
  ggplot(aes(customer_segment, churn_rate, fill = customer_segment)) +
  geom_col() +
  scale_y_continuous(labels = percent) +
  labs(title = "Churn rate by customer segment", x = "Segment", y = "Churn rate") +
  theme_minimal()

# 3.4 distribution of tenure for churned vs not churned
churn %>%
  ggplot(aes(tenure_months, fill = churn)) +
  geom_density(alpha = 0.5) +
  labs(title = "Tenure distribution by churn status", x = "Tenure (months)", y = "Density") +
  theme_minimal()

# customers who churn tend to have shorter tenure, which makes sense

# 3.5 csat score vs churn
churn %>%
  ggplot(aes(churn, csat_score, fill = churn)) +
  geom_boxplot() +
  labs(title = "CSAT score by churn status", x = "Churn", y = "CSAT score") +
  theme_minimal()

# 3.6 nps score vs churn
churn %>%
  ggplot(aes(churn, nps_score, fill = churn)) +
  geom_boxplot() +
  labs(title = "NPS score by churn status", x = "Churn", y = "NPS score") +
  theme_minimal()

# 3.7 correlation matrix of numeric predictors
num_vars <- churn %>% select(where(is.numeric))
corr_mat <- cor(num_vars)
corrplot(corr_mat, method = "color", type = "upper", tl.cex = 0.6)

# nothing too crazy here, no obvious pair of variables that
# are basically duplicates of each other

# 3.8 support tickets vs churn
churn %>%
  group_by(support_tickets) %>%
  summarise(churn_rate = mean(churn == "Yes"), n = n()) %>%
  filter(n > 20) %>%
  ggplot(aes(support_tickets, churn_rate)) +
  geom_col(fill = "steelblue") +
  scale_y_continuous(labels = percent) +
  labs(title = "Churn rate by number of support tickets", x = "Support tickets", y = "Churn rate") +
  theme_minimal()

# ------------------------------------------------------------
# 4. Train / test split
# ------------------------------------------------------------

set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(churn$churn, times = 1, p = 0.2, list = FALSE)
train_set <- churn[-test_index, ]
test_set  <- churn[test_index, ]

# checking that the churn rate is similar in both sets
mean(train_set$churn == "Yes")
mean(test_set$churn == "Yes")

# ------------------------------------------------------------
# 5. Model 1 - Logistic regression
# ------------------------------------------------------------

# logistic regression is a natural starting point for a binary
# classification problem like this one

set.seed(1, sample.kind = "Rounding")

log_fit <- train(
  churn ~ .,
  data = train_set,
  method = "glm",
  family = "binomial"
)

log_pred <- predict(log_fit, test_set)

log_cm <- confusionMatrix(log_pred, test_set$churn, positive = "Yes")
log_cm

log_acc <- log_cm$overall["Accuracy"]
log_f1  <- log_cm$byClass["F1"]
log_sens <- log_cm$byClass["Sensitivity"]
log_spec <- log_cm$byClass["Specificity"]

log_acc
log_f1

# ------------------------------------------------------------
# 6. Model 2 - Random forest
# ------------------------------------------------------------

# random forest should be able to pick up on non-linear
# relationships / interactions between variables that
# logistic regression would miss

set.seed(1, sample.kind = "Rounding")

# using a smaller number of trees and a tuneGrid to keep
# runtime reasonable on a laptop
rf_grid <- expand.grid(mtry = c(2, 5, 8, 12))

rf_fit <- train(
  churn ~ .,
  data = train_set,
  method = "rf",
  tuneGrid = rf_grid,
  ntree = 200,
  importance = TRUE
)

rf_fit$bestTune

rf_pred <- predict(rf_fit, test_set)

rf_cm <- confusionMatrix(rf_pred, test_set$churn, positive = "Yes")
rf_cm

rf_acc <- rf_cm$overall["Accuracy"]
rf_f1  <- rf_cm$byClass["F1"]
rf_sens <- rf_cm$byClass["Sensitivity"]
rf_spec <- rf_cm$byClass["Specificity"]

rf_acc
rf_f1

# variable importance plot
varImpPlot(rf_fit$finalModel, main = "Variable importance - Random Forest")

# ------------------------------------------------------------
# 7. Results comparison
# ------------------------------------------------------------

results <- tibble(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(log_acc, rf_acc),
  Sensitivity = c(log_sens, rf_sens),
  Specificity = c(log_spec, rf_spec),
  F1 = c(log_f1, rf_f1)
)

results

# ------------------------------------------------------------
# End of script
# ------------------------------------------------------------
