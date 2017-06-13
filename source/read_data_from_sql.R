# bowers 2017-06-02
# read data from the donorschoose  database
# library(RPostgreSQL)
library(tidyverse)

my_db <- src_postgres("donorschoose")
projects_tbl <- tbl(my_db, "projects")

projects_tbl %>%
  group_by(funding_status) %>%
  count() %>%
  collect() %>% 
  mutate(funding_status = reorder(funding_status, n)) %>% 
  ggplot(aes(n, funding_status,)) +
    geom_point()

projects_tbl %>%
  group_by(date_posted) %>%
  count() %>%
  arrange(date_posted) %>%
  collect() %>%
  ggplot(aes(date_posted, n)) +
  geom_line() +
  labs(x = "", y = "funded projects per day") 
  ggsave(file = "total_completed_vs_time.pdf", path = "figures/", height = 4)


projects_tbl %>%
  group_by(funding_status, primary_focus_subject) %>%
  filter(funding_status %in% c("completed", "expired")) %>% 
  count() %>%
  collect() %>% 
  mutate(primary_focus_subject = reorder(primary_focus_subject, n)) %>% 
  ggplot(aes(n, primary_focus_subject, shape = funding_status)) +
    geom_point()

projects_tbl %>%
  group_by(funding_status, primary_focus_subject) %>%
  filter(funding_status %in% c("completed", "expired")) %>% 
  count() %>%
  collect() %>% 
#   mutate(primary_focus_subject = reorder(primary_focus_subject, n)) %>% 
  filter(!is.na(primary_focus_subject)) %>% 
  spread(funding_status, n) %>% 
  mutate(frac_funded = completed / (completed + expired)) %>% 
  mutate(primary_focus_subject = reorder(primary_focus_subject, frac_funded)) %>% 
  ggplot(aes(frac_funded, primary_focus_subject)) +
    labs(x = "fraction funded", y = "subject area") +
    geom_point()
    ggsave(file = "fraction_funded_by_subject.pdf", path = "figures")
