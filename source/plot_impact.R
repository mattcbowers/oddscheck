library(tidyverse)
tb <- tibble(donations = c(1, 1.37), type = c("without\nOddsCheck", "with\nOddsCheck"))
theme_set(theme_gray(base_size = 18))

tb %>%
  mutate(type = reorder(type, donations)) %>% 
  ggplot(aes(type, donations, fill = type)) +
  labs(x = "", y = "") +
  guides(fill = FALSE) +
  theme( axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  geom_col()
ggsave(file = "impact.png", path ="../figures/", height = 5, width = 3)
