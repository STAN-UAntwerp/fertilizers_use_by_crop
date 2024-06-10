#Loading libraries
rm(list=ls())
library(terra)      
library(cshapes)    
library(tidyverse)  
library(readxl)
library(exactextractr)
library(readr)
library(ncdf4)
library(sf)
library(ggplot2)
library(stringr)
 

# Loading maps ------------------------------------------------------------
pdf_subfolder <- "pdf_output"
tiff_subfolder <- "tiff_output_final"
#Countries_R_Codes <- read_csv("../maps_output/Countries_R_cshapes.csv")
Countries_codes <- read_excel("../maps_output/Countries_R_Codes.xlsx") %>% rename(Area = 1)
Countries_R_Codes <- read_excel("Other_input/Country_Table.xlsx")
df_start <- read.csv('FUBC_full_dataset_v3.csv')
df_start <- df_start %>%
  select(-c('X', 'Crop_Group', 'Area_report', 'Fert_perc', 'Area_fert', 
            'N_perc_area', 'N_rate', 'N_avg_app', 'P2O5_perc_area', 'P2O5_rate', 
            'P2O5_avg_app', 'K2O_perc_area', 'K2O_rate', 'K2O_avg_app', 
            'NPK_avg_app', 'N_total', 'P2O5_total', 'K2O_total', 'NPK_total', 
            'Source', 'Perc_Agr_Area_FAOStat', 'GDP_Capita_WB', 'GDP_perCapita_UN', 
            'Urea_Price', 'P_rock_Price', 'K2O_Price', 'N_cost', 'P_cost', 'K_cost', 
            'Crop_Global_Price', 'Crop_Price_Nominal', 'Crop_Price_Real', 
            'Country_total_N', 'Country_total_P2O5', 'Country_total_K2O', 
            'N_ha_cropland', 'P2O5_ha_cropland', 'K2O_ha_cropland', 'Area_Irrig', 
            'Edu_Exp', 'pop_pressure', 'Avg_Size_Hold', 'Avg_Size_Hold_stand', 
            'PET', 'MAP', 'TMN', 'AI', 'N_removal', 'P_removal', 'K_removal', 
            'N_removal_ha', 'P_removal_ha', 'K_removal_ha', 'soil_cec', 'soil_ph', 
            'soil_ocs', 'soil_nitrogen', 'soil_clay', 'soil_sand', 'soil_silt', 
            'Region_Name', 'Avg_sqKm', 'Avg_MAP_c', 'Avg_PET_c', 'Avg_TMN_c', 
            'Avg_AI_c', 'Avg_ocs_c', 'Avg_cec_c', 'Avg_ph_c', 'Avg_nitrogen_c', 
            'Avg_clay_c', 'Avg_silt_c', 'Avg_sand_c'))

file_path <- paste0("input/pred_corr_V2.csv")
df_national_bound_1 <- cshp(date = as.Date("1991-01-01"))
df_national_bound_2 <- cshp(date = as.Date("1992-11-11"))
df_national_bound_3 <- cshp(date = as.Date("1993-01-01"))
df_national_bound_4 <- cshp(date = as.Date("2006-10-10"))
df_national_bound_5 <- cshp(date = as.Date("2011-01-01"))
df_national_bound_6 <- cshp(date = as.Date("2012-01-01"))

#Start making maps 
nutrient_elements <- c("N", "K2O", "P2O5")
crop_data_all <- data.frame(
  crop_code = c("1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "3_1", "3_2", "4", "5", "6", "7"),
  crop_name = c("Wheat", "Maize", "Rice", "Other Cereals", "Soybean", "Palm Oil fruit", "Other Oilseeds", "Vegetables", "Fruits", "Roots and tubers", "Sugar crops", "Fiber crops", "Other crops")
)
df_predictions <- read_csv(file_path) %>%
  select(FAOStat_area_code, Year, predicted_N_avg_app, predicted_K2O_avg_app,predicted_P2O5_avg_app, Crop_Code)
df_predictions <- left_join(df_predictions, Countries_R_Codes, by = "FAOStat_area_code")
df_start_unique <- df_start[!duplicated(df_start[c('FAOStat_area_code', 'Crop_Code', 'Year')], fromLast = TRUE), ]
df_predictions_unique <- df_predictions[!duplicated(df_predictions[c('FAOStat_area_code', 'Crop_Code', 'Year')], fromLast = TRUE), ]
df_predictions <- left_join(df_predictions_unique, df_start_unique, by = c('FAOStat_area_code', 'Crop_Code','Year'))

folder_path <- "tiff_output_final"
files <- list.files(folder_path)

for (i in 1:nrow(crop_data_all)) {
  crop <- crop_data_all$crop_code[i]
  crop_name <- crop_data_all$crop_name[i]
  for (year in 1960:1969){
    if (year < 1992) {
      df_national_bound <- df_national_bound_1
    } else if (year == 1992) {
      df_national_bound <- df_national_bound_2
    } else if (year < 2006) {
      df_national_bound <- df_national_bound_3
    } else if (year < 2011) {
      df_national_bound <- df_national_bound_4
    } else if (year < 2012) {
      df_national_bound <- df_national_bound_5
    } else {
      df_national_bound <- df_national_bound_6
    }
    file_path_rast <- paste("../make_maps/input/EQ1_", crop_name, "_", year, ".tiff", sep = "")
    file_var_name <- paste("EQ1_", crop_name, "_", year, sep = "")
    eq_1_maps <- rast(file_path_rast)
    Area_M_Sum <- terra::extract(eq_1_maps,df_national_bound,sum) 
    Area_M_Sum$Area <- df_national_bound$country_name 
    Area_M_Sum <- Area_M_Sum %>% left_join(., Countries_codes, by = "Area") 
    Area_M_Sum <- Area_M_Sum %>%
      select(-c('ID', 'ISO3'))
    df_predictions <- left_join(df_predictions, Area_M_Sum, by = c('FAOStat_area_code'))
    raster_country <- rast(eq_1_maps, vals = 0)
    rasters_boundaries <- list(list())
    for (c in 1:nrow(df_national_bound)){
      Country <- df_national_bound[c,2]
      raster_country_c <- rast(raster_country)
      raster_country_c <- crop(raster_country, Country)
      coverage <- coverage_fraction(raster_country_c, Country)[[1]]
      rasters_boundaries[[c]] <- coverage
      rasters_boundaries[[c]]$country_name <- Country$country_name
    }
    df_predictions_crop_year <- df_predictions %>% filter(Year == year & Crop_Code == crop)
    df_predictions_crop_year$division_result <- df_predictions_crop_year$Area_FAOStat / df_predictions_crop_year[[file_var_name]]
    df_predictions_crop_year$division_result <- ifelse(is.na(df_predictions_crop_year$division_result) 
                                                       | is.infinite(df_predictions_crop_year$division_result), 
                                                       1, df_predictions_crop_year$division_result)
    for (element in nutrient_elements) {
      filename <- paste(crop_name, "_", element, "_", year, ".tiff", sep = "")
      if (filename %in% files) {
        next
      }
      predicted <- paste0("predicted_", element, "_avg_app")
      print(element)
      print(year)
      print(crop_name)
      crop_F_c <- list()
      for (i in 1:length(rasters_boundaries)){
        country_name <- rasters_boundaries[[i]]$country_name[1]$country_name
        if (is.factor(country_name) && length(levels(country_name)) == 1) {
          country_name <- levels(country_name)[1]
        }
        country_name_trimmed <- str_replace_all(country_name, "[[:space:]\\(\\)]", "")
        df_country_year <- df_predictions_crop_year %>% 
          filter(str_replace_all(Cshapes_name, "[[:space:]\\(\\)]", "") == country_name_trimmed)
        eq_1_maps_c <- crop(eq_1_maps, rasters_boundaries[[i]]$lyr.1)
        if (nrow(df_country_year) > 1){
          df_country_year <- df_country_year[1, ]
        }
        if (nrow(df_country_year) != 0){
          eq_1_maps_M <- (eq_1_maps_c * rasters_boundaries[[i]]$lyr.1 * df_country_year$division_result)
          crop_F_c[i] <- (eq_1_maps_M * rasters_boundaries[[i]]$lyr.1 *
                            df_country_year[[predicted]]) / cellSize(eq_1_maps_M, unit = "ha")
        } else {
          eq_1_maps_M <- (eq_1_maps_c * 1)
          crop_F_c[i] <- (eq_1_maps_M * 0)
        }
      }
      crop_F <- crop_F_c[[1]]
      e <- ext(-180, 180, -90, 90)
      crop_F <- extend(crop_F, e)
      for (i in 2:length(crop_F_c)){
        crop_F <-  mosaic(crop_F, crop_F_c[[i]], fun = sum)
      }
      crop_F <- subst(crop_F , NaN, 0)
      save_name_raster <- paste(crop_name, "_", element, "_", year, "_", crop, sep = "")
      names(crop_F)<- save_name_raster
      varnames(crop_F)<- save_name_raster
      tiff_path <- file.path(tiff_subfolder, filename)
      writeRaster(crop_F, tiff_path,overwrite=TRUE)
    }
  }
}


