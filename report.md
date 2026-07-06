# SwiftRoute Logistics — Báo cáo Dự đoán Trễ Giao hàng

## 0. Cấu trúc repo

Repo có **hai notebook**:
- **`coursework.ipynb`** — bản nộp chính, bám sát đúng cấu trúc/yêu cầu của `coursework_requirement.ipynb` (2 model bắt buộc: Logistic Regression + LightGBM, so sánh với random-15%-policy ở Section 8 như đề bài yêu cầu).
- **`experiments.ipynb`** — bản đầy đủ mọi thử nghiệm mở rộng: thêm CatBoost làm model thứ 3, tune trực tiếp theo Recall@Top-15% cho cả 3 model, blend CatBoost+Logistic Regression, ma trận nhầm lẫn chi tiết. Các phần này tự nhận là "beyond the mark scheme" ngay trong notebook.

Báo cáo này mô tả toàn bộ pipeline (áp dụng cho cả hai file ở các phần chung: data, EDA, feature engineering, Model 1/2), và ghi rõ phần nào chỉ có ở `experiments.ipynb`.

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
| `carriers.csv` | 1 dòng/carrier (8 carrier) | `reliability_score`, `capacity_utilisation`, `avg_delay_mins`, `tier_coverage`, `n_active_routes` |
| `weather_events.csv` | 1 dòng/đợt thời tiết xấu | `region`, `event_type` (`snow`/`flood`/`fog`/`storm`/`high_wind`), `severity` (1–3), `start_date`, `end_date` |

Dữ liệu shipment trải dài **900 ngày** (2024-01-01 → 2026-06-19/20).

### 2.2 Chất lượng dữ liệu

- **Missing values**: `weight_kg` thiếu 320 dòng (3.19%), `declared_value` thiếu 507 dòng (5.06%) — đủ thấp để impute (median) thay vì bỏ dòng, xử lý trực tiếp trong pipeline (không phải bước tiền xử lý rời).
- **Trùng lặp**: 0 dòng trùng.
- **Toàn vẹn tham chiếu**: 100% `carrier_id`/`route_id` trong `shipments` khớp với `carriers`/`routes` → join an toàn bằng left-join, không có orphan row.
- **Mất cân bằng lớp**: tỷ lệ trễ tổng thể **11.91%** — không phải lỗi dữ liệu, nhưng là lý do chính chi phối lựa chọn metric và cách xử lý imbalance ở bước huấn luyện.

### 2.3 Join dữ liệu thời tiết — thách thức chính

`weather_events` không join được bằng khóa đơn giản vì mỗi sự kiện là một **khoảng ngày** (`start_date`–`end_date`) theo vùng, còn shipment chỉ có một `pickup_date`. Phải dùng **conditional join (non-equi join)**: với mỗi shipment, tìm sự kiện thời tiết có `region` khớp `origin_region`/`dest_region` và `start_date ≤ pickup_date ≤ end_date`, rồi gộp về severity lớn nhất (`max_weather_severity`) và tổng số sự kiện (`n_weather_events`) để đảm bảo mỗi shipment chỉ có đúng 1 dòng sau join. Cùng logic join đó cũng được tái sử dụng để lấy `event_type` của sự kiện severity cao nhất (Section 4).

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
| `dominant_weather_event_type` | Loại thời tiết (`snow`/`flood`/`fog`/`storm`/`high_wind`/`none`) của sự kiện severity cao nhất ở origin hoặc dest — `severity` (số) coi mọi loại thời tiết cùng mức độ là như nhau, nhưng một trận lũ (`flood`) thực tế có thể chặn đường hoàn toàn theo cách mà bão gió (`storm`) cùng severity không làm được |

Ngoài 11 feature derived này, Section 4.2 cũng join thêm **3 cột raw từ `carriers.csv`** trước đây chưa dùng: `avg_delay_mins` (đo *mức độ nặng* khi carrier trễ, khác `reliability_score` đo *tần suất* trễ), `tier_coverage` (`all`/`express_only`/`standard_economy` — carrier được phép xử lý tier nào), và `n_active_routes` (proxy quy mô/kinh nghiệm carrier).

**Lựa chọn đặc trưng**: dùng ma trận tương quan Pearson (numeric, ngưỡng |r| > 0.85) và Cramér's V (categorical) để phát hiện các cặp feature *diễn giải lại cùng một thông tin*. Quyết định không máy móc bỏ mọi cặp tương quan cao — chỉ bỏ khi một feature thực sự là bản sao/diễn giải lại của feature kia, giữ lại khi chúng phản ánh hai thực tế vận hành khác nhau. `dominant_weather_event_type` và `tier_coverage` đều được kiểm tra Cramér's V — tương quan cao nhất của `tier_coverage` chỉ 0.271 (với `service_tier`), không phải bản sao nên **giữ lại**; `avg_delay_mins`/`n_active_routes` cũng không tương quan >0.85 với cột numeric nào. Kết quả: bỏ 8 feature (`capacity_utilisation`, `avg_transit_days`, `during_weather`, `weather_affected`, `n_weather_events_origin`, `n_weather_events_dest`, `is_monday_or_friday`, `is_cross_region`), còn lại **24 feature** (từ 32 cột candidate).

## 5. Huấn luyện mô hình

### 5.1 Chia train/validation — điểm mấu chốt về phương pháp

Dữ liệu trải dài 900 ngày, nên **không dùng random/stratified split**: một split ngẫu nhiên khiến train và validation phủ cùng một khoảng thời gian, vô tình để mô hình "nhìn thấy tương lai" và chỉ kiểm tra khả năng nội suy chứ không phải khả năng dự báo tiến (forward forecast) — đúng như SwiftRoute cần trong production. Thay vào đó, dữ liệu được sắp theo `pickup_date`, **train trên 80% sớm nhất (8,020 shipment), validate trên 20% gần nhất (2,005 shipment)**. Tương tự, CV dùng `TimeSeriesSplit` (expanding window) thay vì `StratifiedKFold`, để không fold nào validate trên dữ liệu sớm hơn phần train của chính nó.

### 5.2 Hai model bắt buộc (+ CatBoost ở `experiments.ipynb`)

| Model | Cách xử lý imbalance | Regularisation/Tuning |
|---|---|---|
| **Logistic Regression** | `class_weight='balanced'` | L2 (ridge), `C=1.0` — phù hợp vì nhiều feature thời tiết tương quan cao với nhau, L2 co cụm hệ số thay vì zero-out như L1 |
| **LightGBM** | `class_weight='balanced'` | `RandomizedSearchCV` (30 lượt) trên `TimeSeriesSplit`, hội tụ về cây nông, learning rate thấp — hợp lý với ~10k dòng dữ liệu |

*(Chỉ trong `experiments.ipynb`)* **CatBoost** được thêm làm model thứ 3 ngoài yêu cầu, tune tương tự LightGBM, để kiểm tra xem một implementation gradient-boosting khác có đổi kết quả không.

Cả model được huấn luyện trên cả bộ feature đầy đủ và bộ feature đã chọn lọc (Section 4) → 4 tổ hợp kết quả (6 tổ hợp ở `experiments.ipynb` khi tính cả CatBoost).

### 5.3 Kết quả so sánh & mô hình được chọn

| Model (feature set) | Val ROC-AUC |
|---|---|
| Logistic Regression (full features) | 0.8501 |
| Logistic Regression (selected features) | 0.8497 |
| LightGBM (full features) | 0.8447 |
| LightGBM (selected features) | 0.8439 |

*(`experiments.ipynb`: CatBoost đạt 0.8478 full-feature / 0.8460 selected-feature — nằm giữa hai mốc trên, không đổi kết luận.)*

Cả 4 tổ hợp nằm trong khoảng AUC 0.844–0.850 — sát nhau, với một overfit gap thật (~0.02 CV-vs-val) hiện rõ nhờ time-based split. Chênh lệch giữa LR full-feature và LR selected-feature chỉ ~0.0004 — nằm trong nhiễu — nên **mô hình được chọn triển khai là Logistic Regression trên bộ feature đã chọn lọc**: hệ số ổn định hơn, tránh được lệch dấu do đa cộng tuyến. Ví dụ cụ thể: ở bộ full-feature, `reliability_score` mang hệ số **+1.27** — sai dấu, ngụ ý carrier tin cậy hơn lại tăng rủi ro, trái với r=-0.957 tìm thấy ở EDA — vì gần như đối xứng với `capacity_utilisation` (r=-0.97 giữa hai cột). Sau khi bỏ `capacity_utilisation` ở bộ feature đã chọn lọc, hệ số này đảo về **-1.13**, đúng chiều — đúng là loại đa cộng tuyến mà bước chọn lọc feature muốn tránh.

### 5.4 *(Chỉ ở `experiments.ipynb`)* Thử nghiệm hệ thống: tune trực tiếp theo Recall@Top-15%

Cả 3 model (LR, LightGBM, CatBoost) còn được tune lại một lần nữa, lần này chấm điểm trực tiếp bằng **Recall@Top-15%** thay vì ROC-AUC. Kết quả sau khi thêm 3 cột carrier trở nên **không đồng nhất** giữa các model — khác hẳn một lần chạy trước đó (chỉ có `dominant_weather_event_type`, chưa có cột carrier) từng cho thấy cả 3 model hội tụ về đúng kết quả AUC-tuned:
- **Logistic Regression**: tìm ra `C=0.3` khác mặc định `C=1.0`, AUC nhỉnh hơn (0.8500 vs 0.8497) nhưng **Recall@15% giống hệt** (56.2% cả hai cách).
- **LightGBM**: recall-tuning **làm tệ hơn** — chỉ đạt 55.3% so với 57.1% nếu không tune theo recall. Một cảnh báo thực về nhiễu của tín hiệu CV.
- **CatBoost**: recall-tuning **có giúp** — từ 56.6% lên 57.1%.
- Model tốt nhất theo Recall@15%: **LightGBM (AUC-tuned)** và **CatBoost (recall-tuned)** đồng hạng ở **57.1%**, đều vượt mọi biến thể Logistic Regression (55.3–56.2%).

Bài học: việc đổi metric chấm điểm bên trong một search có ngân sách cố định không đảm bảo cải thiện — kết quả phụ thuộc nhiều vào model family và thậm chí thay đổi giữa các lần chạy khi feature set thay đổi.

## 6. Đánh giá mô hình

- **ROC curve**: các tổ hợp model/feature-set đều nằm rõ trên đường chéo random-guess, AUC 0.844–0.850 trên validation set (time-based) — thấp hơn ~0.86–0.87 từng thấy với split ngẫu nhiên trước đây, nhưng là con số **đáng tin hơn** vì phản ánh đúng khả năng dự báo tương lai.
- **Phân phối điểm dự đoán**: nhóm `is_delayed=1` lệch rõ về bên phải so với nhóm `is_delayed=0`, có overlap thật (hợp lý ở mức AUC ~0.85) — phần lớn delay thật nằm trên ngưỡng top-15%, đây chính là lý do capture rate ở Section business impact cao.
- **Hệ số Logistic Regression** (top ảnh hưởng, model đã triển khai): `origin_weather_severity` (+1.68), `dest_weather_severity` (+1.36), `reliability_score` (-1.13, đúng chiều), `service_tier_express` (+0.62), `is_winter` (+0.57), `dominant_weather_event_type_high_wind` (-0.35 — một sự kiện gió mạnh gắn với rủi ro *thấp hơn* một trận lũ cùng mức severity, hợp lý vì lũ có thể chặn đường hoàn toàn), `weekend_buffer_risk` (+0.29), `weather_both_ends` (+0.23). Đáng chú ý: **3 cột carrier mới** (`avg_delay_mins`, `tier_coverage`, `n_active_routes`) **không lọt top-15** — chúng qua được bước lọc redundancy nhưng không mang thêm tín hiệu biên đáng kể ngoài những gì `reliability_score`/`historical_delay_rate` đã nắm bắt. Một ngoại lệ: `max_weather_severity` mang dấu âm (-1.20) dù về logic phải dương — **artefact đa cộng tuyến dư** (trùng lặp thông tin với 2 cột severity origin/dest), không ảnh hưởng AUC nhưng hệ số này không nên đọc theo nghĩa đen.

## 7. Tác động kinh doanh (Business Impact)

Trên tập validation (2,005 shipment gần nhất, 226 delay thật), chính sách **flag top 15% theo rủi ro dự đoán** (301 shipment) bắt được:

- **Model policy: 127/226 delay — capture rate 56.2%** — tiết kiệm **£5,715** (giả định £45/delay tránh được)
- **Random 15% policy** (đối chứng, theo đúng yêu cầu đề bài): chỉ **33/226 (14.6%)**, tiết kiệm **£1,485**
- **Lift: 3.85x**, tương đương **£4,230 giá trị tăng thêm** so với chọn ngẫu nhiên, trên slice validation này

**Ma trận nhầm lẫn tại đúng ngưỡng top-15%** (không phải ngưỡng 0.5 mặc định):

| | Không flag | Flag (top 15%) |
|---|---|---|
| **Không trễ** | TN = 1,605 | FP = 174 |
| **Trễ** | FN = 99 | TP = 127 |

→ Precision = 42.2%, Recall = 56.2%, Macro F1 = 0.702. Precision dưới 50% có nghĩa hơn một nửa shipment bị flag hóa ra vẫn đúng hẹn — nhưng đây là hệ quả tất yếu của trần can thiệp 15%: mô hình buộc phải flag đúng 301 shipment bất kể có bao nhiêu shipment thực sự rủi ro cao.

**Khuyến nghị**: thay bất kỳ cách chọn thủ công/ngẫu nhiên nào bằng risk ranking của mô hình. Với cùng ngân sách 15%, Operations bắt được khoảng 5–6/10 delay thật thay vì ~1.5/10 — mô hình không tốn thêm chi phí vận hành, chỉ giúp nhắm đúng năng lực can thiệp sẵn có vào đúng shipment cần nó.

## 8. *(Chỉ ở `experiments.ipynb`)* Thử nghiệm mở rộng: Blend CatBoost + Logistic Regression

Recall 56.2% vẫn để lọt gần một nửa số delay thật. Thử blend trung bình có trọng số giữa xác suất dự đoán của LR và CatBoost (recall-tuned), trọng số tune qua CV theo `TimeSeriesSplit`. Kết quả: trọng số tối ưu `w=0.70` (70% LR, 30% CatBoost) đạt CV Recall@15% 62.1% — nhưng trên validation set thật, blend ở trọng số này lại cho **đúng 127/226 (56.2%)**, **trùng khớp tuyệt đối với LR một mình**, không cải thiện gì so với baseline, dù ROC-AUC tổng thể của blend (0.8508) cao hơn cả hai thành phần.

**Kết luận trung thực**: CV báo hiệu tích cực nhưng validation set không xác nhận — Recall@Top-15% là hàm bậc thang, tín hiệu CV ở quy mô dữ liệu này (~1,300 dòng/fold) quá nhiễu để đảm bảo chuyển hóa thành lợi ích thật. Nếu mục tiêu duy nhất là tối đa hóa Recall@Top-15%, CatBoost hoặc LightGBM một mình (57.1%, Section 5.4) hiện đang vượt mọi phương án có Logistic Regression tham gia — nhưng lý do interpretability là tại sao Logistic Regression, không phải model cây, được lưu vào `model.pkl`.

## 9. Giám sát drift khi lên production

- **Input drift**: so sánh phân phối từng feature theo cửa sổ trượt (tuần) với phân phối lúc train — PSI hoặc Kolmogorov-Smirnov cho numeric (`weight_kg`, `distance_km`, `max_weather_severity`), chi-square cho categorical (`service_tier`, `carrier_id`). Nhóm dễ drift nhất: mix `service_tier`/`carrier_id` (do quyết định thương mại của SwiftRoute) và các feature thời tiết (mùa vụ tự nhiên). Ngưỡng cảnh báo: PSI > 0.2 hoặc p-value có ý nghĩa thống kê **và** effect size đủ lớn (tránh báo động giả do cỡ mẫu lớn).
- **Target drift**: theo dõi hàng tuần (1) tỷ lệ trễ thực tế so với baseline ~11.9%, (2) ROC-AUC/calibration thực tế so với baseline ~0.85 (đo được vài ngày sau khi có kết quả giao hàng thật). Kích hoạt retrain khi: AUC live giảm rõ rệt và kéo dài nhiều tuần, capture rate ở ngưỡng 15% giảm, hoặc có thay đổi cấu trúc mới (carrier mới, vùng mới) mà mô hình chưa từng thấy.

## 10. Extension

- **E1 — Phân cụm tuyến đường**: KMeans (3 cụm) trên `distance_km`, `n_depot_hops`, `historical_delay_rate`, `avg_transit_days` cho ra một cụm "long-haul" rõ rệt (22 tuyến, TB 532km, 3 hop, **24.4%** delay rate lịch sử) so với cụm "short-haul" (16 tuyến, TB 161km, 1 hop, **10.4%**) — rủi ro hơn gấp 2 lần. Cụm long-haul tập trung quanh London, Wales, South_East, Scotland — khớp với các tuyến/carrier rủi ro nhất đã thấy ở EDA. Kết luận: rủi ro của mạng lưới SwiftRoute mang tính **cấu trúc** (tuyến dài + nhiều hop), tập trung ở vài hành lang cụ thể chứ không rải đều.
- **E2 — SHAP**: giải thích 10 shipment rủi ro cao nhất (predicted risk 98.8–99.9%) — `origin_weather_severity`/`dest_weather_severity` luôn là 2 yếu tố đóng góp lớn nhất, tiếp theo là `reliability_score` thấp. Đây là các trường hợp thời tiết xấu + carrier yếu cộng dồn. `max_weather_severity` vẫn cho SHAP âm dù shipment rủi ro cao — cùng artefact đa cộng tuyến đã nêu ở Section 6.

## 11. Hướng đã cân nhắc nhưng không áp dụng

**SMOTE** (oversampling tổng hợp lớp thiểu số) được cân nhắc để tăng recall nhưng quyết định không dùng, vì ba lý do: (1) `class_weight='balanced'` đã xử lý đúng vấn đề imbalance ở bước huấn luyện, dùng thêm SMOTE dễ over-correct mà không cộng dồn lợi ích; (2) metric quan trọng là recall tại **top-15% theo rank xác suất**, không phải tại ngưỡng 0.5 cố định — SMOTE chủ yếu dịch chuyển ngưỡng quyết định, ít tác động đến thứ tự xếp hạng tương đối giữa các shipment; (3) rủi ro kỹ thuật: nội suy SMOTE trên các cột nhị phân/one-hot dễ tạo sample vô nghĩa (trừ khi dùng SMOTENC), và phải áp dụng trong từng fold của `TimeSeriesSplit` để tránh leak — thêm độ phức tạp cho lợi ích không chắc chắn.

## 12. Tổng kết

- Mô hình triển khai: **Logistic Regression (bộ feature đã chọn lọc, 24 feature)**, Val ROC-AUC ≈ 0.850, lưu tại `model.pkl`.
- Tại ngưỡng vận hành thực tế (top 15%): capture rate 56.2% (127/226), tiết kiệm ước tính £5,715 — lift 3.85x so với chọn ngẫu nhiên (£4,230 giá trị tăng thêm).
- Đã rà soát toàn bộ cột dữ liệu sẵn có: thêm `dominant_weather_event_type` và 3 cột carrier — tất cả vượt qua kiểm tra redundancy, nhưng 3 cột carrier không mang thêm tín hiệu biên đáng kể. Muốn tăng recall thêm cần **nguồn dữ liệu mới**, không phải tune/kết hợp model nhiều hơn trên feature hiện tại (đã thử ở `experiments.ipynb`, kết quả không đồng nhất/không đáng tin cậy).
- Toàn bộ đánh giá dùng **time-based split**, nên các con số trên là ước lượng dự báo tiến (forward-looking) thực tế, không phải nội suy lạc quan trong cùng một giai đoạn thời gian.
