library(tidyverse)

my_db <- src_postgres("donorschoose")
projects_tbl <- tbl(my_db, "projects")

projects <- projects_tbl %>%
  filter(date_posted > "2015-01-01", date_posted < "2015-06-01")  %>% 
  collect()

# Trim out missing values and unneeded columns
projects_trim <- projects  %>% 
  select(funding_status, school_zip, school_metro, school_charter, school_magnet, school_year_round, school_nlns, school_kipp, school_charter_ready_promise, teacher_prefix, teacher_teach_for_america, teacher_ny_teaching_fellow, primary_focus_area, secondary_focus_subject, secondary_focus_area, resource_type, poverty_level, grade_level, vendor_shipping_charges, sales_tax, payment_processing_charges, total_price_excluding_optional_support, students_reached)  
prop_missing <- colMeans(is.na(projects_trim)) 
high_miss_cols <- prop_missing[prop_missing > 0.05]
projects_trim <- projects_trim[ , !(names(projects_trim) %in% names(high_miss_cols))]
projects_trim <- na.omit(projects_trim)
projects_trim <- projects_trim  %>% 
  filter(total_price_excluding_optional_support < 20000)  %>% 
  filter(funding_status %in% c("completed", "expired"))


# Modeling -------------------------------------------------------------
projects_trim$funding_status <- as.factor(projects_trim$funding_status)
# saveRDS(projects_trim, "data_rds/projects_trim_sample.rds")
projects_trim <- readRDS("data_rds/projects_trim_sample.rds")
mod0 <- glm(funding_status ~ 1, data = projects_trim, family = "binomial")
mod_full <- glm(funding_status ~ ., data = projects_trim, family = binomial)
mod_price <- glm(funding_status ~ total_price_excluding_optional_support, data = projects_trim, family = "binomial")
# positive coeficients indicate increased probability of expiration.

# Best subset selection
library(leaps)
mod_bestsub <- regsubsets(funding_status ~ ., data = projects_trim, nvmax = 40)
mod_bestsub_sum <- summary(mod_bestsub)
names(mod_bestsub_sum)
plot(mod_bestsub)
plot(mod_bestsub_sum$bic)
which.min(mod_bestsub_sum$bic)
coef(mod_bestsub, 16)
# 16 (lowest so far)

# Forward Selection
mod_forward <- regsubsets(funding_status ~ ., data = projects_trim, method = "forward", nvmax = 19)
mod_forward_sum <- summary(mod_forward)
plot(mod_forward)
plot(mod_forward_sum$bic)
which.min(mod_forward_sum$bic)

# Backward Selection
mod_backward <- regsubsets(funding_status ~ ., data = projects_trim, method = "backward", nvmax = 19)
mod_backward_sum <- summary(mod_backward)
plot(mod_backward)
plot(mod_backward_sum$bic)
which.min(mod_backward_sum$bic)

## Ridge and Lasso
library(glmnet)
X <- model.matrix(funding_status ~ ., data = projects_trim)
Y <- projects_trim$funding_status
mod_ridge <- glmnet(X, Y, family = "binomial", alpha = 0)
ridge_cv <- cv.glmnet(X, Y, family = "binomial", alpha = 0)
plot(ridge_cv)
plot(mod_ridge, xvar = "lambda", label = TRUE)
mod_lasso <- glmnet(X, Y, family = "binomial", alpha = 1)
lasso_cv <- cv.glmnet(X, Y, family = "binomial", alpha = 1)
plot(lasso_cv)
plot(mod_lasso, xvar = "lambda", label = TRUE)

# Find the linearly dependent rows--------------------------------------
# turns out to be subject/area
# Why is the model matrix not full rank?
# library(Matrix)
# X <- model.matrix(mod_full)
# dim(X)
# rankMatrix(X)
# qq <- qr(X)
# indep <- X[,qq$pivot[seq(qq$rank)]]
# dep <- X[, -qq$pivot[seq(qq$rank)]]
# dimnames(indep)
# dimnames(dep)
# str(dep)
# colSums(dep)
# xtabs(~ primary_focus_subject + primary_focus_area, data = projects)
# # drop primary_focus_area because it is linearly dependent with subject
