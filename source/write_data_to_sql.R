# bowers 2017-06-02
# write donorschoose data to a postgres database
library(RPostgreSQL)
library(tidyverse)
library(readr)

database_name <- "donorschoose"

drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, 
  host = "localhost", 
  user = "postgres", 
  password = "", 
  dbname = database_name
)

# write the csv files to postgres tables
write_to_postgres <- function(table_name, col_names){
  files <- system(paste0("ls data_csv/", table_name), intern = TRUE)
  ## write first table
  tb <- read_csv(paste0("data_csv/", table_name, "/", files[[1]]), col_names = col_names)
  dbWriteTable(con, 
    name = table_name, 
    value = tb,
    overwrite = TRUE,
    row.names = FALSE
  )
  # Append data
  for (i in 2:length(files)){
    tb <- read_csv(paste0("data_csv/", table_name, "/", files[[i]]), col_names = col_names)
    dbWriteTable(con, 
      name = table_name, 
      value = tb,
      append = TRUE,
      row.names = FALSE
    )
  }
  dbDisconnect(con)
}

write_to_postgres <- function(table_name, col_names){
  files <- system(paste0("ls data_csv/", table_name), intern = TRUE)
  ## write first table
#   tb <- read_csv(paste0("data_csv/", table_name, "/", files[[1]]), col_names = col_names)
  tb <- read_delim(paste0("data_csv/", table_name, "/", files[[1]]), col_names = col_names, delim = ",", escape_backslash = TRUE, escape_double = FALSE)
  dbDisconnect(con)
  return(tb)
}

# table_name <- "projects"
# col_names <- c("_projectid", "_teacher_acctid", "_schoolid", "school_ncesid", "school_latitude", "school_longitude", "school_city", "school_state", "school_zip", "school_metro", "school_district", "school_county", "school_charter", "school_magnet", "school_year_round", "school_nlns", "school_kipp", "school_charter_ready_promise", "teacher_prefix", "teacher_teach_for_america", "teacher_ny_teaching_fellow", "primary_focus_subject", "primary_focus_area" ,"secondary_focus_subject", "secondary_focus_area", "resource_type", "poverty_level", "grade_level", "vendor_shipping_charges", "sales_tax", "payment_processing_charges", "fulfillment_labor_materials", "total_price_excluding_optional_support", "total_price_including_optional_support", "students_reached", "total_donations", "num_donors", "eligible_double_your_impact_match", "eligible_almost_home_match", "funding_status", "date_posted", "date_completed", "date_thank_you_packet_mailed", "date_expiration")
# table_name <- "donations"
# col_names <- c("_donationid", "_projectid", "_donor_acctid", "_cartid", "donor_city", "donor_state", "donor_zip", "is_teacher_acct", "donation_timestamp", "donation_to_project", "donation_optional_support", "donation_total", "donation_included_optional_support", "payment_method", "payment_included_acct_credit", "payment_included_campaign_gift_card", "payment_included_web_purchased_gift_card", "payment_was_promo_matched", "is_teacher_referred", "giving_page_id", "giving_page_type", "for_honoree", "thank_you_packet_mailed")
# table_name <- "resources"
# col_names <- c("_resourceid", "_projectid", "vendorid", "vendor_name", "item_name", "item_number", "item_unit_price", "item_quantity")
# table_name <- "giving_pages"
# col_names <- c("giving_page_id", "_creator_acctid", "created_date", "is_active", "most_recent_donation", "amount_raised", "number_of_donors", "number_of_students", "number_of_projects_supported", "number_of_teachers", "number_of_schools")
# table_name <- "giving_page_projects"
# col_names <- c("giving_page_id", "_projectid")
table_name <- "essays"
col_names <- c("_projectid", "_teacherid", "title", "short_description", "need_statement", "essay", "thankyou_note", "impact_letter")

write_to_postgres(table_name, col_names)
