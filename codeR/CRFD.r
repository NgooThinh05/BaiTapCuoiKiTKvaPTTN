# =========================================================================
# 4. THÍ NGHIỆM CRFD (TƯƠNG TÁC GIỮA k VÀ max_depth)
# =========================================================================
cat("\n=======================================================\n")
cat("--- 4.1 ĐANG HUẤN LUYỆN THÍ NGHIỆM CRFD (Sẽ mất 1-2 phút) ---\n")
cat("=======================================================\n")

# Định nghĩa các yếu tố
k_values <- c(3, 5, 10)
depth_values <- c(3, 5, 0) # 0 tương đương với Unlimited (không giới hạn)
crfd_results <- list()

# Chạy vòng lặp cho 9 tổ hợp (3 mức k  x  3 mức max_depth)
for (k_val in k_values) {
  for (d_val in depth_values) {
    set.seed(1234)
    folds_crfd <- vfold_cv(mlc_churn_clean, v = k_val, repeats = 10, strata = churn)
    
    # Cấu hình RF với max.depth thay đổi
    rf_spec_crfd <- rand_forest(mode = "classification") %>% 
      set_engine("ranger", max.depth = d_val, seed = 1234, num.threads = 1)
    
    rf_wf_crfd <- workflow() %>% 
      add_model(rf_spec_crfd) %>% 
      add_formula(churn ~ .)
    
    # Huấn luyện (Lưu ý: Sẽ xuất hiện cảnh báo Precision NA khi depth=3, điều này là bình thường do underfitting)
    res_crfd <- fit_resamples(rf_wf_crfd, resamples = folds_crfd, metrics = f1_metric)
    
    f1_scores <- collect_metrics(res_crfd, summarize = FALSE) %>% filter(.metric == "f_meas")
    depth_label <- ifelse(d_val == 0, "Unlimited", as.character(d_val))
    
    crfd_results[[paste0("k", k_val, "_d", depth_label)]] <- data.frame(
      k = as.factor(k_val),
      max_depth = factor(depth_label, levels = c("3", "5", "Unlimited")),
      F1 = f1_scores$.estimate
    )
  }
}

crfd_data <- bind_rows(crfd_results)
write.csv(crfd_data, "CRFD_Results.csv", row.names = FALSE)
cat("\n>> Đã lưu toàn bộ kết quả CRFD vào file: CRFD_Results.csv\n")

# -------------------------------------------------------------------------
# PHÂN TÍCH THỐNG KÊ VÀ VẼ ĐỒ THỊ CRFD
# -------------------------------------------------------------------------
cat("\n>> 4.2 BẢNG THỐNG KÊ MÔ TẢ CHO CÁC TỔ HỢP CRFD:\n")
crfd_summary <- crfd_data %>%
  group_by(k, max_depth) %>%
  summarise(
    N_mau = n(),
    Trung_binh_F1 = mean(F1, na.rm = TRUE),
    Std = sd(F1, na.rm = TRUE),
    `CI_95%(+/-)` = qt(0.975, df = n() - 1) * (sd(F1, na.rm = TRUE) / sqrt(n())),
    .groups = "drop"
  )
print(as.data.frame(crfd_summary), digits = 4)

cat("\n>> 4.3 PHÂN TÍCH PHƯƠNG SAI HAI CHIỀU (TWO-WAY ANOVA):\n")
aov_crfd <- aov(F1 ~ k * max_depth, data = crfd_data)
print(summary(aov_crfd))

cat("\n>> 4.4 PHÂN TÍCH BẰNG lm() LẤY TRUNG BÌNH & KHOẢNG TIN CẬY 95%:\n")
crfd_data$group <- interaction(crfd_data$k, crfd_data$max_depth)
lm_crfd <- lm(F1 ~ group - 1, data = crfd_data)
print(cbind(Mean = coef(lm_crfd), confint(lm_crfd)))

