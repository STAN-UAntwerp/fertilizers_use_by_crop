# Loading libraries -------------------------------------------------------
library(readr)            #For loading csv
library(tidyverse)        #For managing data

# Loading data ------------------------------------------------------------
N_predictions <- read_csv("~/crop_database_imputation/crop_database_imputation/results_corrected/N_avg_app/HGB/full_predictions_merged.csv")
P2O5_predictions <- read_csv("~/crop_database_imputation/crop_database_imputation/results_corrected/P2O5_avg_app/HGB/full_predictions_merged.csv")
K2O_predictions <- read_csv("~/crop_database_imputation/crop_database_imputation/results_corrected/K2O_avg_app/HGB/full_predictions_merged.csv")

Ludemann_2023_grasslands <- read_csv("~/crop_database_imputation/crop_database_imputation/OV_databases/Ludemann_2023_grasslands.csv")

# DB Correction Creation --------------------------------------------------
## They are duplicated because the original db has more than one data in three cases for the same crop, year, country combination
DB_Correction <- N_predictions %>% select(FAOStat_area_code, Year, Crop_Code, Country_total_N, Area_FAOStat, predicted_N_avg_app) %>% 
  left_join(., P2O5_predictions %>% select(FAOStat_area_code, Year, Crop_Code, Country_total_P2O5, predicted_P2O5_avg_app), by = c("Year", "Crop_Code", "FAOStat_area_code"))%>% 
  left_join(., K2O_predictions %>% select(FAOStat_area_code, Year, Crop_Code, Country_total_K2O, predicted_K2O_avg_app), by = c("Year", "Crop_Code", "FAOStat_area_code")) %>% unique()

## Adding the percentage use for grasslands
DB_Correction <- DB_Correction %>% left_join(., Ludemann_2023_grasslands %>% select(FAOStat_area_code, Perc_N_grass, Perc_P2O5_grass, Perc_K2O_grass), by = "FAOStat_area_code") %>% 
  mutate(Perc_N_grass = ifelse(is.na(Perc_N_grass), 1, Perc_N_grass), Perc_P2O5_grass = ifelse(is.na(Perc_P2O5_grass), 1, Perc_P2O5_grass), Perc_K2O_grass = ifelse(is.na(Perc_K2O_grass), 1, Perc_K2O_grass))

## Estimating the total use without grasslands
DB_Correction <- DB_Correction %>% mutate(Country_total_N_nograss = Country_total_N * Perc_N_grass, Country_total_P2O5_nograss = Country_total_P2O5 * Perc_P2O5_grass, Country_total_K2O_nograss = Country_total_K2O * Perc_K2O_grass)


# Predictions corrected ---------------------------------------------------
## First we estimate the totals with our predictions
Totals_predictions <- DB_Correction %>% mutate(N_total_pred = Area_FAOStat*predicted_N_avg_app, P2O5_total_pred = Area_FAOStat*predicted_P2O5_avg_app, K2O_total_pred = Area_FAOStat*predicted_K2O_avg_app) %>%
  group_by(FAOStat_area_code, Year) %>% summarise(N_total_pred = sum(N_total_pred, na.rm = T)/1000, P2O5_total_pred = sum(P2O5_total_pred, na.rm = T)/1000, K2O_total_pred = sum(K2O_total_pred, na.rm = T)/1000)

## Second, we estimate the relationship between the totals, and the country totals
Totals_rel <- Totals_predictions %>% left_join(., DB_Correction %>% select(FAOStat_area_code, Year, Country_total_N_nograss, Country_total_P2O5_nograss, Country_total_K2O_nograss) %>% unique(), by = c("FAOStat_area_code", "Year")) %>%
  mutate(rel_N_totals = Country_total_N_nograss/N_total_pred, rel_P2O5_totals = Country_total_P2O5_nograss/P2O5_total_pred,rel_K2O_totals = Country_total_K2O_nograss/K2O_total_pred)

## Third, estimate the predictions corrected
Predictions_corrected <- DB_Correction %>% select(FAOStat_area_code, Crop_Code, Year, predicted_N_avg_app, predicted_P2O5_avg_app, predicted_K2O_avg_app) %>% left_join(., Totals_rel, by = c("FAOStat_area_code", "Year")) %>% 
  mutate(predicted_N_avg_app_cor = predicted_N_avg_app*rel_N_totals, predicted_P2O5_avg_app_cor = predicted_P2O5_avg_app*rel_P2O5_totals, predicted_K2O_avg_app_cor = predicted_K2O_avg_app*rel_K2O_totals)

Predictions_corrected_db <- Predictions_corrected %>% select(FAOStat_area_code, Crop_Code, Year, predicted_N_avg_app_cor, predicted_K2O_avg_app_cor, predicted_P2O5_avg_app_cor)
write.csv(Predictions_corrected_db, "~/crop_database_imputation/crop_database_imputation/prediction_corrected_byTotals/Prediction_corrected.csv")

# Plots and others --------------------------------------------------------

## Check the NA
Predictions_corrected_db %>% filter(is.na(predicted_N_avg_app_cor)) %>% count()
N_predictions %>% filter(is.na(Country_total_N)) %>% count()
## All the NAs seem to be because a NA in the totals of the country in the original db

Predictions_corrected_db %>% filter(is.na(predicted_P2O5_avg_app_cor)) %>% count()
P2O5_predictions %>% filter(is.na(Country_total_P2O5)) %>% count()
## All the NAs seem to be because a NA in the totals of the country in the original db

Predictions_corrected_db %>% filter(is.na(predicted_K2O_avg_app_cor)) %>% count()
K2O_predictions %>% filter(is.na(Country_total_K2O)) %>% count()
## All the NAs seem to be because a NA in the totals of the country in the original db

## Plots
Predictions_corrected_reg <- Predictions_corrected %>% left_join(., N_predictions %>% select(FAOStat_area_code, Region_Name) %>% unique(), by = "FAOStat_area_code")

p1 <- ggplot(Predictions_corrected_reg, aes(x = predicted_N_avg_app, y = predicted_N_avg_app_cor)) +
  geom_point(alpha = 0.1) + geom_abline(slope =1, intercept = 1, color = "red") + facet_wrap(~Region_Name) + ylim(0,800)  + theme_bw() +
  theme(strip.text = element_text(face = "bold", color = "black", hjust = 0),
        strip.background = element_rect(fill = "white", linetype = "solid", color = "transparent", linewidth = 1)) +
    ylab("Corrected N average application HGB prediction") + xlab("N average application HGB prediction")

p2 <- ggplot(Predictions_corrected_reg, aes(x = predicted_P2O5_avg_app, y = predicted_P2O5_avg_app_cor)) +
  geom_point(alpha = 0.1) + geom_abline(slope =1, intercept = 1, color = "red") + facet_wrap(~Region_Name) + ylim(0,400)  + theme_bw() +
  theme(strip.text = element_text(face = "bold", color = "black", hjust = 0),
        strip.background = element_rect(fill = "white", linetype = "solid", color = "transparent", linewidth = 1)) +
  ylab(expression("Corrected"*P["2"]*O["5"]*" average application HGB prediction")) + xlab(expression(P["2"]*O["5"]*" average application HGB prediction"))

p3 <- ggplot(Predictions_corrected_reg, aes(x = predicted_P2O5_avg_app, y = predicted_P2O5_avg_app_cor)) +
  geom_point(alpha = 0.1) + geom_abline(slope =1, intercept = 1, color = "red") + facet_wrap(~Region_Name) + ylim(0,500)  + theme_bw() +
  theme(strip.text = element_text(face = "bold", color = "black", hjust = 0),
        strip.background = element_rect(fill = "white", linetype = "solid", color = "transparent", linewidth = 1)) +
  ylab(expression("Corrected"*K["2"]*O*" average application HGB prediction")) + xlab(expression(K["2"]*O*" average application HGB prediction"))

png(file = 'Plots/N_Regions_pred_vs_corrected.png',width = 600, height = 600)
p1
dev.off()

png(file = 'Plots/P2O5_Regions_pred_vs_corrected.png',width = 600, height = 600)
p2
dev.off()

png(file = 'Plots/K2O_Regions_pred_vs_corrected.png',width = 600, height = 600)
p3
dev.off()


