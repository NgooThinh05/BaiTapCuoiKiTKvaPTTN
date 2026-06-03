# 📊 Dự án: Dự đoán Khách hàng Rời mạng (Churn Prediction) bằng Random Forest


## 📌 Tổng quan dự án
Dự án này ứng dụng các phương pháp **Thiết kế và Phân tích Thực nghiệm (DoE)** để đánh giá và tối ưu hóa hiệu năng của mô hình học máy Random Forest trong việc dự đoán tỷ lệ khách hàng rời mạng viễn thông. Hệ thống phân tích được xây dựng trên ngôn ngữ **R**, tập trung vào việc kiểm soát sai số ngẫu nhiên, phân tích phương sai (ANOVA) và đánh giá tác động tương tác của các siêu tham số.

## 🗄️ Dữ liệu sử dụng
Dự án sử dụng bộ dữ liệu `mlc_churn` thu thập từ một công ty viễn thông tại Mỹ.
* **Quy mô:** 5000 mẫu dữ liệu với 20 cột đặc trưng.
* **Đặc điểm phân phối:** Dữ liệu mất cân bằng nghiêm trọng với 85.86% khách hàng ở lại (no) và 14.14% khách hàng rời mạng (yes).
* **Tiền xử lý:**
    * Loại bỏ 4 biến cước phí (`charge`) do có tương quan tuyến tính hoàn hảo (r = 1.0) với các biến số phút gọi (`minutes`) nhằm tránh hiện tượng đa cộng tuyến (multicollinearity).
    * Loại bỏ các biến mang tính cục bộ cao (`state`, `area_code`) để hạn chế rủi ro quá khớp (overfitting).

---

## 🔬 Các thực nghiệm cốt lõi

### 1. Mô hình Cơ sở (Mô hình M)
* **Thiết lập:** Cấu hình Random Forest mặc định (không giới hạn độ sâu cây), cố định hạt giống ngẫu nhiên (`seed = 1234`), và thiết lập đơn luồng (`num.threads = 1`) để loại bỏ nhiễu hệ thống. Đánh giá bằng phương pháp Stratified 5-Fold lặp 10 lần.
* **Kết quả:** Mô hình đạt F1-Score ~0.81. Ba đặc trưng mang tính quyết định nhất gồm: `total_day_minutes`, `number_customer_service_calls`, và `total_eve_minutes`.

### 2. Thí nghiệm Ngẫu nhiên Hoàn toàn (CRD)
* **Mục tiêu:** Khảo sát ảnh hưởng của số lượng fold huấn luyện (`k` = 3, 5, 10).
* **Phân tích thống kê:** * Kiểm định Levene xác nhận dữ liệu đồng nhất phương sai (p = 0.07486 > 0.05).
    * Phân tích One-way ANOVA khẳng định số fold `k` tác động có ý nghĩa thống kê đến hiệu năng (p = 0.0262 < 0.05).
    * Phân tích hậu định Tukey HSD chứng minh cấu hình k=10 vượt trội hơn hẳn so với k=3.

### 3. Thí nghiệm Đa yếu tố (CRFD)
* **Mục tiêu:** Đánh giá ảnh hưởng đồng thời và sự tương tác giữa số lượng fold (`k`) và độ sâu tối đa của cây (`max_depth` = 3, 5, Unlimited).
* **Phân tích thống kê:**
    * Two-way ANOVA chứng minh cả 2 yếu tố tác động cực kỳ mạnh mẽ (p < 2e-16). Yếu tố `max_depth` là biến số quyết định sinh tử.
    * **Hiệu ứng tương tác:** Tồn tại sự tương tác có ý nghĩa thống kê giữa `k` và `max_depth` (p = 0.00412). Lượng dữ liệu lớn (k=10) chỉ thực sự phát huy sức mạnh khi thuật toán có đủ độ phức tạp (max_depth = 5 hoặc Unlimited) để học các đặc trưng.

---

## 🛠️ Công cụ & Thư viện sử dụng
* **Ngôn ngữ:** R
* **Framework:** `tidymodels` (Xây dựng pipeline Machine Learning)
* **Packages thống kê:** `car` (Levene's Test), `stats` (ANOVA, Tukey HSD)
* **Trực quan hóa:** `ggplot2`, `ggcorrplot`, `vip`
