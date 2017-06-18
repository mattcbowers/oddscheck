library(tidyverse)
library(forcats)

tb <- read_csv("data_pkl/coef_lasso.csv", col_names = c("coef", "var")) %>%
  filter(coef != 0) %>%
  mutate(var = reorder(var, -coef)) %>%
  mutate(sig = factor(sign(coef)))  %>%
  mutate(var = fct_recode(var,
    "Total Project Price" = "total_price_excluding_optional_support",
    "Resource: Technology" = "resource_type_Technology",
    "Teacher: 'Mrs.'" = "teacher_prefix_Mrs.",
    "State: CA" = "school_state_CA",
    "Resource: Books" = "resource_type_Books",
    "High Poverty" = "poverty_level_highest poverty"
  ))

theme_set(theme_gray(base_size = 18))
tb %>%
  ggplot(aes(var, coef, fill = sig))+
  geom_col() +
  guides(fill = FALSE) +
  labs(x = "", y = "coefficient") +
  coord_flip()

ggsave(file = "coef_lasso.png", path = "../figures", height = 3)
