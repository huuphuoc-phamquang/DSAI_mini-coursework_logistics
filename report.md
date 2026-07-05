# SwiftRoute Logistics — Báo cáo Dự đoán Trễ Giao hàng

## 1. Bài toán

SwiftRoute Logistics muốn dự đoán **rủi ro trễ giao hàng ngay tại thời điểm lấy hàng (pickup)**, để đội Operations có thể can thiệp chủ động (đổi tuyến, đổi carrier, cảnh báo khách hàng...) trước khi shipment thực sự bị trễ. Đây là bài toán **phân loại nhị phân** với nhãn `is_delayed` (1 = trễ, 0 = đúng hẹn), mỗi dòng dữ liệu là một shipment.

Ràng buộc nghiệp vụ quan trọng nhất: đội Operations **chỉ có năng lực can thiệp trên 15% shipment rủi ro cao nhất**. Điều này quyết định toàn bộ cách chọn metric và cách đánh giá mô hình xuyên suốt project — không dùng accuracy (vì chỉ ~12% shipment bị trễ, một mô hình "luôn đoán không trễ" đã đạt ~88% accuracy nhưng vô dụng), mà dùng:
- **ROC-AUC** để so sánh/chọn mô hình (không phụ thuộc ngưỡng, đo khả năng xếp hạng đúng).
- **Recall/Precision/F1 tại đúng ngưỡng top-15%** (không phải ngưỡng mặc định 0.5) để đánh giá giá trị kinh doanh thực tế, vì đó là ngưỡng Operations thực sự dùng.

## 2. Dữ liệu

### 2.1 Các bảng nguồn

| File | Vai trò | Số cột chính |
|---|---|---|
| `shipments.csv` | Bảng chính, 1 dòng/shipment | `pickup_date`, `origin_region`, `dest_region`, `service_tier`, `carrier_id`, `route_id`, `weight_kg`, `volume_cm3`, `declared_value`, `is_business`, `is_delayed` |
| `routes.csv` | 1 dòng/tuyến origin–destination | `distance_km`, `n_depot_hops`, `historical_delay_rate`, `avg_transit_days` |
| `carriers.csv` | 1 dòng/carrier (8 carrier) | `reliability_score`, `capacity_utilisation`, `avg_delay_mins`, `n_active_routes`, `tier_coverage` |
| `weather_events.csv` | 1 dòng/đợt thời tiết xấu | `region`, `event_type` (`snow`/`flood`/`fog`/`storm`/`high_wind`), `severity` (1–3), `start_date`, `end_date` |

Dữ liệu shipment trải dài **900 ngày** (2024-01-01 → 2026-06-19/20).

### 2.2 Chất lượng dữ liệu

- **Missing values**: `weight_kg` thiếu 320 dòng (3.19%), `declared_value` thiếu 507 dòng (5.06%) — đủ thấp để impute (median) thay vì bỏ dòng, xử lý trực tiếp trong pipeline (không phải bước tiền xử lý rời).
- **Trùng lặp**: 0 dòng trùng.
- **Toàn vẹn tham chiếu**: 100% `carrier_id`/`route_id` trong `shipments` khớp với `carriers`/`routes` → join an toàn bằng left-join, không có orphan row.
- **Mất cân bằng lớp**: tỷ lệ trễ tổng thể **11.91%** — không phải lỗi dữ liệu, nhưng là lý do chính chi phối lựa chọn metric và cách xử lý imbalance ở bước huấn luyện.

### 2.3 Join dữ liệu thời tiết — thách thức chính

`weather_events` không join được bằng khóa đơn giản vì mỗi sự kiện là một **khoảng ngày** (`start_date`–`end_date`) theo vùng, còn shipment chỉ có một `pickup_date`. Phải dùng **conditional join (non-equi join)**: với mỗi shipment, tìm sự kiện thời tiết có `region` khớp `origin_region`/`dest_region` và `start_date ≤ pickup_date ≤ end_date`, rồi gộp về severity lớn nhất (`max_weather_severity`) và tổng số sự kiện (`n_weather_events`) để đảm bảo mỗi shipment chỉ có đúng 1 dòng sau join (tránh nhân bản do quan hệ 1-nhiều). Cùng logic join đó cũng được tái sử dụng để lấy `event_type` của sự kiện severity cao nhất (Section 4).

## 3. Khám phá dữ liệu (EDA) — các phát hiện chính

Xếp theo mức độ liên quan đến trễ giao hàng:

1. **Độ tin cậy carrier** là tín hiệu mạnh nhất: reliability_score của carrier tương quan với tỷ lệ trễ ở mức **r = -0.957**. BudgetShip (27.7% trễ) là điểm nghẽn hệ thống, xuất hiện lặp lại ở top các tuyến rủi ro nhất (Scotland→South_East 71.4%, London→Scotland 60%, Wales→London 58%...), trong khi FastTrack chỉ 3.8% — đây là **vấn đề ở cấp carrier, không phải ở cấp tuyến đường**.
2. **Thời tiết**: shipment lấy hàng trong thời gian có sự kiện thời tiết xấu trễ **29.8%** so với **8.6%** khi không có (lift ~3.5x), ảnh hưởng 15.5% tổng số shipment.
3. **Khoảng cách tuyến**: tương quan dương vừa phải (r = 0.214), tỷ lệ trễ tăng từ 4.7% (0–200km) lên 26.0% (600–800km).
4. **Service tier**: express trễ nhiều nhất (16.6%) so với economy (9.6%) — phản trực giác vì express là tier cao cấp, nhưng hợp lý vì cửa sổ thời gian giao hàng của express hẹp hơn, ít dư địa hấp thụ gián đoạn.
5. **Tính mùa vụ**: tỷ lệ trễ tháng 12/1 luôn cao vượt trội so với trung bình năm (~11.9%) — ví dụ Dec 2025 = 21.5%, Jan 2025 = 19.0% — trực tiếp gợi ý feature `is_winter`.
6. **`weight_kg`** lệch phải mạnh (skew = 2.07; mean 5.06kg vs median 3.45kg).

## 4. Feature engineering

Từ bảng join gốc, xây dựng thêm **11 feature mới**, mỗi feature bám theo một phát hiện cụ thể từ EDA (không chọn ngẫu nhiên):

| Feature | Ý nghĩa |
|---|---|
| `weather_affected` | Có sự kiện thời tiết ở origin HOẶC dest |
| `weather_both_ends` | Cả origin VÀ dest cùng bị ảnh hưởng thời tiết — rủi ro cộng dồn |
| `is_winter` | Pickup rơi vào tháng 11–2, bám theo tính mùa vụ ở EDA |
| `volume_to_weight` | Tỷ trọng volume/weight — proxy cho hàng cồng kềnh/bất thường |
| `carrier_route_risk` | `(1 - reliability_score) × historical_delay_rate` — carrier kém + tuyến rủi ro cao |
| `day_of_week` / `is_monday_or_friday` | Chu kỳ workload đầu/cuối tuần ở depot |
| `is_cross_region` | Shipment liên vùng hay nội vùng |
| `n_weather_events` | Tổng số sự kiện thời tiết chồng lên ngày pickup (origin+dest) |
| `implied_speed_required` | `distance_km / số ngày cam kết theo tier` — áp lực thời gian trên quãng đường |
| `weekend_buffer_risk` | Express + pickup thứ Sáu → deadline rơi vào thứ Bảy, thời điểm depot thiếu nhân sự (delay rate 16.1% vs 11.7%) |
| `dominant_weather_event_type` | Loại thời tiết (`snow`/`flood`/`fog`/`storm`/`high_wind`/`none`) của sự kiện severity cao nhất ở origin hoặc dest — `severity` (số) coi mọi loại thời tiết cùng mức độ là như nhau, nhưng một trận lũ (`flood`) thực tế có thể chặn đường hoàn toàn theo cách mà bão gió (`storm`) cùng severity không làm được, nên một cờ phân loại cho phép model học hiệu ứng riêng theo từng loại |

**Lựa chọn đặc trưng**: dùng ma trận tương quan Pearson (numeric, ngưỡng |r| > 0.85) và Cramér's V (categorical) để phát hiện các cặp feature *diễn giải lại cùng một thông tin* (ví dụ `avg_transit_days` ≈ `distance_km`/tốc độ trung bình, `weather_affected` bị bao hàm bởi `n_weather_events > 0`). Quyết định không máy móc bỏ mọi cặp tương quan cao — chỉ bỏ khi một feature thực sự là bản sao/diễn giải lại của feature kia, giữ lại khi chúng phản ánh hai thực tế vận hành khác nhau (ví dụ `distance_km` và `n_depot_hops` tương quan 0.90 nhưng vẫn giữ cả hai). `dominant_weather_event_type` được kiểm tra Cramér's V với các cột categorical khác — tương quan cao nhất chỉ 0.266 (với `weather_both_ends`), không phải bản sao nên được **giữ lại**. Kết quả: bỏ 8 feature (`capacity_utilisation`, `avg_transit_days`, `during_weather`, `weather_affected`, `n_weather_events_origin`, `n_weather_events_dest`, `is_monday_or_friday`, `is_cross_region`), còn lại 21 feature (từ 29).

## 5. Huấn luyện mô hình

### 5.1 Chia train/validation — điểm mấu chốt về phương pháp

Dữ liệu trải dài 900 ngày, nên **không dùng random/stratified split**: một split ngẫu nhiên khiến train và validation phủ cùng một khoảng thời gian, vô tình để mô hình "nhìn thấy tương lai" và chỉ kiểm tra khả năng nội suy chứ không phải khả năng dự báo tiến (forward forecast) — đúng như SwiftRoute cần trong production. Thay vào đó, dữ liệu được sắp theo `pickup_date`, **train trên 80% sớm nhất (8,020 shipment), validate trên 20% gần nhất (2,005 shipment)**. Tương tự, CV dùng `TimeSeriesSplit` (expanding window) thay vì `StratifiedKFold`, để không fold nào validate trên dữ liệu sớm hơn phần train của chính nó.

### 5.2 Ba mô hình được thử

| Model | Cách xử lý imbalance | Regularisation/Tuning |
|---|---|---|
| **Logistic Regression** | `class_weight='balanced'` | L2 (ridge), `C=1.0` — phù hợp vì nhiều feature thời tiết tương quan cao với nhau, L2 co cụm hệ số thay vì zero-out như L1 |
| **LightGBM** | `class_weight='balanced'` | `RandomizedSearchCV` (30 lượt) trên `TimeSeriesSplit`, hội tụ về cây nông, learning rate thấp — hợp lý với ~10k dòng dữ liệu |
| **CatBoost** (thêm ngoài yêu cầu) | `auto_class_weights='Balanced'` | Cùng cách tune như LightGBM, dùng chung `build_preprocessor()`/`evaluate_model()` để so sánh trực tiếp |

Cả 3 model được huấn luyện trên cả bộ feature đầy đủ và bộ feature đã chọn lọc (Section 4) → 6 tổ hợp kết quả.

### 5.3 Kết quả so sánh & mô hình được chọn

| Model (feature set) | Val ROC-AUC |
|---|---|
| Logistic Regression (full features) | 0.8510 |
| Logistic Regression (selected features) | 0.8504 |
| CatBoost (selected features) | 0.8473 |
| CatBoost (full features) | 0.8468 |
| LightGBM (full features) | 0.8460 |
| LightGBM (selected features) | 0.8454 |

Cả 6 tổ hợp đều nằm trong khoảng AUC 0.845–0.851 — sát nhau, với một overfit gap thật (~0.02 CV-vs-val) hiện rõ nhờ time-based split (split ngẫu nhiên trước đó đã che giấu gap này). Chênh lệch giữa LR full-feature và LR selected-feature chỉ ~0.0005 — nằm trong nhiễu — nên **mô hình được chọn triển khai là Logistic Regression trên bộ feature đã chọn lọc**: hệ số ổn định hơn, tránh được lệch dấu do đa cộng tuyến. Ví dụ cụ thể: ở bộ full-feature, `reliability_score` mang hệ số **+1.12** — sai dấu, ngụ ý carrier tin cậy hơn lại tăng rủi ro, trái với r=-0.957 tìm thấy ở EDA — vì gần như đối xứng với `capacity_utilisation` (r=-0.97 giữa hai cột). Sau khi bỏ `capacity_utilisation` ở bộ feature đã chọn lọc, hệ số này đảo về **-1.13**, đúng chiều — đúng là loại đa cộng tuyến mà bước chọn lọc feature muốn tránh.

### 5.4 Thử nghiệm hệ thống: tune trực tiếp theo Recall@Top-15%

Ngoài việc chọn model theo ROC-AUC (5.3), cả 3 model còn được **tune lại một lần nữa trên cùng bộ feature đã chọn lọc**, nhưng lần này `RandomizedSearchCV`/`GridSearchCV` được chấm điểm trực tiếp bằng **Recall@Top-15%** (hàm `recall_at_top_k`) thay vì ROC-AUC — vì đây mới là con số Operations thực sự dùng.

Phát hiện đáng chú ý: việc đổi tiêu chí tune **không tạo ra model khác** ở cả 3 trường hợp — mỗi bản recall-tuned trùng khớp tuyệt đối với bản AUC-tuned tương ứng (cả Val Recall@15% lẫn Val AUC). Với Logistic Regression, lý do đơn giản: search chọn ra đúng `C=1.0` — giá trị mặc định vốn đã dùng ở 5.3 (5.3 chưa từng tune `C` theo AUC). Với LightGBM/CatBoost, hai lượt search (AUC và recall) dùng chung `random_state` nên duyệt đúng 30 bộ hyperparameter giống hệt nhau — ứng viên thắng theo AUC hóa ra cũng thắng (hoặc đồng hạng) theo Recall@15% trong chính 30 ứng viên đó. Bảng so sánh cuối cùng vì vậy được quyết định gần như hoàn toàn bởi **loại model**, không phải bởi tiêu chí tune: **CatBoost đạt 58.4% Val Recall@15%** (AUC 0.847), vượt rõ Logistic Regression/LightGBM (55.6–56.2%), bất kể tune theo mục tiêu nào. Đây là kết quả có giá trị dù kém kịch tính: đổi metric chấm điểm bên trong một search có ngân sách cố định (`n_iter=30`) không đảm bảo tìm ra kết quả khác — cần một lưới hyperparameter khác hẳn hoặc ngân sách lớn hơn mới thực sự khai thác được sự đánh đổi AUC-vs-recall.

Model triển khai vẫn là **Logistic Regression (bộ feature đã chọn lọc)** vì lý do interpretability ở 5.3, dù CatBoost cho kết quả tốt hơn trên đúng metric vận hành.

## 6. Đánh giá mô hình

- **ROC curve**: cả 6 tổ hợp đều nằm rõ trên đường chéo random-guess, AUC 0.845–0.851 — thấp hơn con số ~0.86–0.87 từng thấy với split ngẫu nhiên trước đây, nhưng là con số **đáng tin hơn** vì phản ánh đúng khả năng dự báo tương lai.
- **Phân phối điểm dự đoán**: nhóm `is_delayed=1` lệch rõ về bên phải so với nhóm `is_delayed=0`, có overlap thật (hợp lý ở mức AUC ~0.85) — phần lớn delay thật nằm trên ngưỡng top-15%, đây chính là lý do capture rate ở Section business impact cao.
- **Hệ số Logistic Regression** (top ảnh hưởng, model đã triển khai): `origin_weather_severity` (+1.67), `dest_weather_severity` (+1.36), `reliability_score` (-1.13, đúng chiều), `service_tier_express` (+0.59), `is_winter` (+0.57). Feature mới nhất `dominant_weather_event_type` cũng mang tín hiệu thật: so với nhóm nền (`flood`), một sự kiện `high_wind` (-0.36) hay `storm` (-0.17) gắn với rủi ro **thấp hơn** ở cùng mức severity số — hợp lý vì lũ lụt có thể chặn đường hoàn toàn theo cách bão gió cùng mức độ không làm được. `weekend_buffer_risk` (+0.28) và `weather_both_ends` (+0.20) cũng có tín hiệu thật. Một ngoại lệ: `max_weather_severity` mang dấu âm (-1.20) dù về logic phải là dấu dương — đây là **artefact đa cộng tuyến dư** (trùng lặp thông tin với 2 cột severity theo origin/dest), không ảnh hưởng đến khả năng xếp hạng (AUC) nhưng hệ số này không nên đọc theo nghĩa đen.

## 7. Tác động kinh doanh (Business Impact)

Trên tập validation (2,005 shipment gần nhất, 226 delay thật), chính sách **flag top 15% theo rủi ro dự đoán** (301 shipment) bắt được:

- **126/226 delay — capture rate 55.8%**
- Tiết kiệm SLA: **£5,670** (giả định £45/delay tránh được)

**Ma trận nhầm lẫn tại đúng ngưỡng top-15%** (không phải ngưỡng 0.5 mặc định):

| | Không flag | Flag (top 15%) |
|---|---|---|
| **Không trễ** | TN = 1,604 | FP = 175 |
| **Trễ** | FN = 100 | TP = 126 |

→ Precision = 41.9%, Recall = 55.8%, Macro F1 = 0.700. Precision dưới 50% có nghĩa hơn một nửa shipment bị flag hóa ra vẫn đúng hẹn — nhưng đây là hệ quả tất yếu của trần can thiệp 15%: mô hình buộc phải flag đúng 301 shipment bất kể có bao nhiêu shipment thực sự rủi ro cao, nên một phần flag chắc chắn là báo động giả.

**Khuyến nghị**: dùng risk ranking của mô hình để quyết định shipment nào Operations can thiệp. Với cùng ngân sách 15%, Operations bắt được khoảng 5–6/10 delay thật — mô hình không tốn thêm chi phí vận hành, chỉ giúp nhắm đúng năng lực can thiệp sẵn có vào đúng shipment cần nó.

## 8. Các thử nghiệm mở rộng để tăng recall — và một phát hiện trung thực

Recall 55.8% ở model chính vẫn để lọt gần một nửa số delay thật, nên đã thử 2 hướng cải thiện thêm (dựa trên kết quả tune ở 5.4), cả hai đều đánh giá qua CV trước, chỉ áp dụng lên validation set một lần cuối để tránh nhìn trước dữ liệu:

**8.1 — Có nên tune trực tiếp theo Recall@Top-15% không?** Vì bản LR "recall-tuned" ở 5.4 trùng khớp tuyệt đối với bản mặc định (cùng `C=1.0`), câu trả lời thực nghiệm ở đây là **không tạo khác biệt gì**: cả hai đều bắt **126/226 (55.8%)**, tiết kiệm cùng **£5,670** — chênh lệch **£0**. Đây là kết quả gọn nhưng thực chất: lưới 9 giá trị `C` được thử không chứa một điểm regularisation nào tốt hơn cho mục tiêu này trên bộ feature hiện tại, vì ranking của Logistic Regression đã bị chi phối bởi vài tín hiệu mạnh (thời tiết, độ tin cậy carrier), ít dư địa để `C` xáo trộn lại top 15%.

**8.2 — Blend CatBoost + Logistic Regression**: hai mô hình xếp hạng shipment khác nhau (LR tuyến tính vs CatBoost bắt tương tác phi tuyến), nên trung bình có trọng số hai xác suất dự đoán có thể đẩy thêm vài delay thật vào top 15%. Trọng số blend được tune qua CV theo từng fold của `TimeSeriesSplit`. Kết quả CV cho thấy trọng số tối ưu `w=0.80` (80% LR, 20% CatBoost) đạt CV Recall@15% 61.9% — nhưng trên validation set thật, blend ở trọng số này lại cho **đúng 126/226 (55.8%)**, **trùng khớp tuyệt đối với LR một mình** và **không cải thiện gì** so với baseline. Lý do: trọng số 80% LR quá thiên về LR, không đủ để CatBoost đẩy bất kỳ shipment nào qua ngưỡng top-301 — tín hiệu CV tốt đã không "sống sót" khi áp dụng lên validation set.

**Kết luận trung thực**: cả 2 thử nghiệm mở rộng đều **không cải thiện** được recall so với model gốc lần này — khác với một lần chạy trước (trước khi thêm `dominant_weather_event_type`) từng cho thấy cải thiện nhỏ. Điều này tự nó là một phát hiện có giá trị: Recall@Top-15% là một hàm bậc thang, tín hiệu CV để chọn hyperparameter/trọng số blend khá nhiễu ở quy mô dữ liệu này (~1.300 dòng/fold), nên một cải thiện nhỏ tìm được qua CV không đảm bảo tái lập trên validation set — không phải một lợi ích cấu trúc đáng tin cậy. Nếu mục tiêu duy nhất là tối đa hóa Recall@Top-15%, **CatBoost một mình** (58.4%, Section 5.4) hiện đang vượt mọi phương án có Logistic Regression tham gia (kể cả blend) — nhưng lý do interpretability ở 5.3 là lý do Logistic Regression, không phải CatBoost, được lưu vào `model.pkl`.

## 9. Giám sát drift khi lên production

- **Input drift**: so sánh phân phối từng feature theo cửa sổ trượt (tuần) với phân phối lúc train — PSI hoặc Kolmogorov-Smirnov cho numeric (`weight_kg`, `distance_km`, `max_weather_severity`), chi-square cho categorical (`service_tier`, `carrier_id`). Nhóm dễ drift nhất: mix `service_tier`/`carrier_id` (do quyết định thương mại của SwiftRoute) và các feature thời tiết (mùa vụ tự nhiên). Ngưỡng cảnh báo: PSI > 0.2 hoặc p-value có ý nghĩa thống kê **và** effect size đủ lớn (tránh báo động giả do cỡ mẫu lớn).
- **Target drift**: theo dõi hàng tuần (1) tỷ lệ trễ thực tế so với baseline ~11.9%, (2) ROC-AUC/calibration thực tế so với baseline ~0.85 (đo được vài ngày sau khi có kết quả giao hàng thật). Kích hoạt retrain khi: AUC live giảm rõ rệt và kéo dài nhiều tuần, capture rate ở ngưỡng 15% giảm, hoặc có thay đổi cấu trúc mới (carrier mới, vùng mới) mà mô hình chưa từng thấy.

## 10. Extension

- **E1 — Phân cụm tuyến đường**: KMeans (3 cụm) trên `distance_km`, `n_depot_hops`, `historical_delay_rate`, `avg_transit_days` cho ra một cụm "long-haul" rõ rệt (22 tuyến, TB 532km, 3 hop, **24.4%** delay rate lịch sử) so với cụm "short-haul" (16 tuyến, TB 161km, 1 hop, **10.4%**) — rủi ro hơn gấp 2 lần. Cụm long-haul tập trung quanh London, Wales, South_East, Scotland — khớp với các tuyến/carrier rủi ro nhất đã thấy ở EDA. Kết luận: rủi ro của mạng lưới SwiftRoute mang tính **cấu trúc** (tuyến dài + nhiều hop), tập trung ở vài hành lang cụ thể chứ không rải đều.
- **E2 — SHAP**: giải thích 10 shipment rủi ro cao nhất (predicted risk 98.6–99.9%) — `origin_weather_severity`/`dest_weather_severity` luôn là 2 yếu tố đóng góp lớn nhất, tiếp theo là `reliability_score` thấp. Đây là các trường hợp thời tiết xấu + carrier yếu cộng dồn. `max_weather_severity` vẫn cho SHAP âm dù shipment rủi ro cao — cùng artefact đa cộng tuyến đã nêu ở Section 6.

## 11. Hướng đã cân nhắc nhưng không áp dụng

**SMOTE** (oversampling tổng hợp lớp thiểu số) được cân nhắc để tăng recall nhưng quyết định không dùng, vì ba lý do: (1) `class_weight='balanced'` đã xử lý đúng vấn đề imbalance ở bước huấn luyện, dùng thêm SMOTE dễ over-correct mà không cộng dồn lợi ích; (2) metric quan trọng là recall tại **top-15% theo rank xác suất**, không phải tại ngưỡng 0.5 cố định — SMOTE chủ yếu dịch chuyển ngưỡng quyết định, ít tác động đến thứ tự xếp hạng tương đối giữa các shipment; (3) rủi ro kỹ thuật: nội suy SMOTE trên các cột nhị phân/one-hot dễ tạo sample vô nghĩa (trừ khi dùng SMOTENC), và phải áp dụng trong từng fold của `TimeSeriesSplit` để tránh leak — thêm độ phức tạp cho lợi ích không chắc chắn.

## 12. Tổng kết

- Mô hình triển khai: **Logistic Regression (bộ feature đã chọn lọc, 21 feature)**, Val ROC-AUC ≈ 0.850, lưu tại `model.pkl`.
- Tại ngưỡng vận hành thực tế (top 15%): capture rate 55.8%, tiết kiệm ước tính £5,670 trên slice validation ~2,000 shipment.
- Hai thử nghiệm mở rộng (tune trực tiếp theo Recall@15%, blend CatBoost+LR) **không** cho thấy cải thiện nào lần này — một phát hiện trung thực về sự nhiễu của tín hiệu CV cho một metric dạng bậc thang, hơn là một thất bại của phương pháp. CatBoost một mình vẫn là lựa chọn tốt nhất thuần theo Recall@15% (58.4%), nhưng không được chọn triển khai vì ưu tiên interpretability.
- Toàn bộ đánh giá dùng **time-based split**, nên các con số trên là ước lượng dự báo tiến (forward-looking) thực tế, không phải nội suy lạc quan trong cùng một giai đoạn thời gian.
