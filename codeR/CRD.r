# =========================================================================
# CÀI ĐẶT VÀ TẢI THƯ VIỆN
# =========================================================================
library(tidymodels)
library(modeldata)
library(car)      # Cho leveneTest()
library(ggplot2)  # Để vẽ đồ thị
library(dplyr)    # Để xử lý bảng thống kê mô tả

# =========================================================================
# 1. TIỀN XỬ LÝ DỮ LIỆU
# =========================================================================
data(mlc_churn, package = "modeldata")

mlc_churn_clean <- mlc_churn %>%
  select(-state, -area_code, -ends_with("charge")) %>%
  mutate(churn = factor(churn, levels = c("yes", "no")))

f1_metric <- metric_set(f_meas)

# =========================================================================
# 3. THÍ NGHIỆM CRD (ĐÁNH GIÁ ẢNH HƯỞNG CỦA SỐ FOLD k)
# =========================================================================
cat("\n=======================================================\n")
cat("--- 3.1 ĐANG HUẤN LUYỆN THÍ NGHIỆM CRD (k = 3, 5, 10) ---\n")
cat("=======================================================\n")

k_values <- c(3, 5, 10)
crd_results <- list()

rf_spec_crd <- rand_forest(mode = "classification") %>% 
  set_engine("ranger", seed = 1234, num.threads = 1)

for (k_val in k_values) {
  set.seed(1234)
  folds_crd <- vfold_cv(mlc_churn_clean, v = k_val, repeats = 10, strata = churn)
  
  rf_wf_crd <- workflow() %>% 
    add_model(rf_spec_crd) %>% 
    add_formula(churn ~ .)
  
  res_crd <- fit_resamples(rf_wf_crd, resamples = folds_crd, metrics = f1_metric)
  
  f1_scores <- collect_metrics(res_crd, summarize = FALSE) %>% filter(.metric == "f_meas")
  crd_results[[paste0("k", k_val)]] <- data.frame(k = as.factor(k_val), F1 = f1_scores$.estimate)
}

crd_data <- bind_rows(crd_results)
write.csv(crd_data, "CRD_Results.csv", row.names = FALSE)

# -------------------------------------------------------------------------
# XUẤT KẾT QUẢ 
# -------------------------------------------------------------------------
cat("\n>> 3.2 KẾT QUẢ MÔ TẢ THỐNG KÊ:\n")
crd_summary <- crd_data %>%
  group_by(k) %>%
  summarise(
    N_mau = n(),
    Trung_binh_F1 = mean(F1, na.rm = TRUE),
    Std = sd(F1, na.rm = TRUE),
    `CI_95%(+/-)` = qt(0.975, df = n() - 1) * (sd(F1, na.rm = TRUE) / sqrt(n())),
    Min = min(F1, na.rm = TRUE),
    Max = max(F1, na.rm = TRUE)
  )
print(as.data.frame(crd_summary), digits = 4)

cat("\n>> 3.3 KIỂM ĐỊNH PHƯƠNG SAI - LEVENE TEST:\n")
levene_crd <- leveneTest(F1 ~ k, data = crd_data)
print(levene_crd)

cat("\n>> 3.4 PHÂN TÍCH PHƯƠNG SAI MỘT CHIỀU - ONE-WAY ANOVA:\n")
aov_crd <- aov(F1 ~ k, data = crd_data)
print(summary(aov_crd))

cat("\n>> 3.5 SO SÁNH BỘI - TUKEY HSD:\n")
tukey_crd <- TukeyHSD(aov_crd)
print(tukey_crd)

# Vẽ đồ thị Tukey
plot(tukey_crd, col = "darkblue", las = 1)


