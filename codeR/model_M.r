# =========================================================================
# CÀI ĐẶT VÀ TẢI THƯ VIỆN
# =========================================================================
# Bỏ ghi chú dòng dưới để cài đặt nếu máy bạn chưa có:
# install.packages(c("tidymodels", "modeldata", "vip", "ggplot2"))

library(tidymodels)
library(modeldata)
library(vip)      # Để vẽ đồ thị Tầm quan trọng đặc trưng
library(ggplot2)  # Để tinh chỉnh đồ thị

# =========================================================================
# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# =========================================================================
cat("\n--- 1. ĐANG TIỀN XỬ LÝ DỮ LIỆU ---\n")
data(mlc_churn, package = "modeldata")

mlc_churn_clean <- mlc_churn %>%
  # Loại bỏ biến đa cộng tuyến và biến nhiễu vị trí
  select(-state, -area_code, -ends_with("charge")) %>%
  # Đặt "yes" làm lớp tích cực để tính chuẩn xác F1-Score
  mutate(churn = factor(churn, levels = c("yes", "no")))

# Đặt chỉ số đo lường là F1-Score
f1_metric <- metric_set(f_meas)

# =========================================================================
# 2. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH M (RANDOM FOREST)
# =========================================================================
cat("\n--- 2. ĐANG HUẤN LUYỆN MÔ HÌNH M BẰNG CROSS-VALIDATION ---\n")

# Cố định bộ sinh số ngẫu nhiên để chia fold không đổi
set.seed(1234)
folds_M <- vfold_cv(mlc_churn_clean, v = 5, repeats = 10, strata = churn)

# Cấu hình RF: Cố định seed, tắt đa luồng (num.threads = 1) và bật importance
rf_spec_M <- rand_forest(mode = "classification") %>% 
  set_engine("ranger", importance = "impurity", seed = 1234, num.threads = 1)

rf_wf_M <- workflow() %>% 
  add_model(rf_spec_M) %>% 
  add_formula(churn ~ .)

# Đánh giá F1-Score trên các tập CV
res_M <- fit_resamples(
  rf_wf_M, 
  resamples = folds_M, 
  metrics = f1_metric
)

# In kết quả rút gọn từ collect_metrics()
f1_M_summary <- collect_metrics(res_M) %>% filter(.metric == "f_meas")
cat("\n>> Kết quả F1-Score trung bình (rút gọn) của Mô hình M:\n")
print(f1_M_summary)

# =========================================================================
# 3. PHÂN TÍCH HIỆU NĂNG BẰNG HÀM lm()
# =========================================================================
cat("\n--- 3. KẾT QUẢ PHÂN TÍCH MÔ HÌNH M BẰNG lm() ---\n")

# Trích xuất toàn bộ 50 điểm F1-Score từ 5 fold x 10 lần lặp
f1_scores_M <- collect_metrics(res_M, summarize = FALSE) %>% 
  filter(.metric == "f_meas")

# Sử dụng công thức .estimate ~ 1 để lấy trung bình và khoảng tin cậy của toàn bộ mô hình
lm_M <- lm(.estimate ~ 1, data = f1_scores_M)

cat("\n>> Bảng tóm tắt hồi quy lm() cho F1-Score:\n")
print(summary(lm_M))

cat("\n>> Khoảng tin cậy 95% của F1-Score:\n")
print(confint(lm_M))

