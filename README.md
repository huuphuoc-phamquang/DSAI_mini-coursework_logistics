# SwiftRoute Logistics - Báo cáo Dự đoán Trễ Giao hàng

## 0. Cấu trúc repo

Repo có **hai notebook**:
- **`coursework.ipynb`**: File báo cáo chính thức, tuân thủ nghiêm ngặt cấu trúc và các tiêu chí đánh giá trong `coursework_requirement.ipynb`. Nội dung tập trung vào hai mô hình Logistic Regression và LightGBM.
- **`experiments.ipynb`**: File nghiên cứu mở rộng, thử nghiệm thêm mô hình thứ ba (CatBoost), tối ưu hóa trực tiếp theo chỉ số Recall@Top-15% cho cả ba mô hình, kết hợp kỹ thuật blend (CatBoost + Logistic Regression) và phân tích chi tiết ma trận nhầm lẫn (confusion matrix).

Tài liệu này trình bày toàn bộ quy trình xây dựng pipeline (áp dụng chung cho cả hai file ở các phần: xử lý dữ liệu, phân tích EDA, kỹ nghệ đặc trưng và hai mô hình cơ bản), đồng thời chú thích rõ ràng các nội dung phân tích nâng cao chỉ xuất hiện ở file `experiments.ipynb`.

## 1. Bài toán

SwiftRoute Logistics muốn dự đoán **rủi ro trễ giao hàng ngay tại thời điểm lấy hàng (pickup)**, để đội Operations có thể can thiệp chủ động (đổi tuyến, đổi carrier, cảnh báo khách hàng...) trước khi shipment thực sự bị trễ. Đây là bài toán **phân loại nhị phân** với nhãn `is_delayed` (1 = trễ, 0 = đúng hẹn), mỗi dòng dữ liệu là một shipment.

Ràng buộc nghiệp vụ quan trọng nhất: đội Operations **chỉ có năng lực can thiệp trên 15% shipment rủi ro cao nhất**. Điều này quyết định toàn bộ cách chọn metric và cách đánh giá mô hình xuyên suốt project - không dùng accuracy (vì chỉ ~12% shipment bị trễ, một mô hình "luôn đoán không trễ" đã đạt ~88% accuracy nhưng vô dụng), mà dùng:
- **ROC-AUC** để so sánh/chọn mô hình (không phụ thuộc ngưỡng, đo khả năng xếp hạng đúng).
- **Recall/Precision/F1 tại đúng ngưỡng top-15%** (không phải ngưỡng mặc định 0.5) để đánh giá giá trị kinh doanh thực tế, vì đó là ngưỡng Operations thực sự dùng.

## 2. Dữ liệu

### 2.1 Các bảng nguồn

| File | Vai trò | Số cột chính |
|---|---|---|
| `shipments.csv` | Bảng chính, 1 dòng/shipment | `pickup_date`, `origin_region`, `dest_region`, `service_tier`, `carrier_id`, `route_id`, `weight_kg`, `volume_cm3`, `declared_value`, `is_business`, `is_delayed` |
| `routes.csv` | 1 dòng/tuyến origin - destination | `distance_km`, `n_depot_hops`, `historical_delay_rate`, `avg_transit_days` |
| `carriers.csv` | 1 dòng/carrier (8 carrier) | `reliability_score`, `capacity_utilisation`, `avg_delay_mins`, `tier_coverage`, `n_active_routes` |
| `weather_events.csv` | 1 dòng/đợt thời tiết xấu | `region`, `event_type` (`snow`/`flood`/`fog`/`storm`/`high_wind`), `severity` (1-3), `start_date`, `end_date` |

Dữ liệu shipment trải dài **900 ngày** (2024-01-01 → 2026-06-19/20).

### 2.2 Chất lượng dữ liệu

- **Missing values**: `weight_kg` thiếu 320 dòng (3.19%), `declared_value` thiếu 507 dòng (5.06%) - đủ thấp để impute (median) thay vì bỏ dòng, xử lý trực tiếp trong pipeline (không phải bước tiền xử lý rời).
- **Trùng lặp**: 0 dòng trùng.
- **Toàn vẹn tham chiếu**: 100% `carrier_id`/`route_id` trong `shipments` khớp với `carriers`/`routes` → join an toàn bằng left-join, không có orphan row.
- **Mất cân bằng lớp**: tỷ lệ trễ tổng thể **11.91%** - không phải lỗi dữ liệu, nhưng là lý do chính chi phối lựa chọn metric và cách xử lý imbalance ở bước huấn luyện.

### 2.3 Join dữ liệu thời tiết - thách thức chính

Việc liên kết dữ liệu với bảng `weather_events` không thể thực hiện bằng các khóa nối thông thường do sự khác biệt về định dạng thời gian: mỗi sự kiện thời tiết kéo dài trong một **khoảng ngày** (`start_date` – `end_date`) theo từng khu vực, trong khi mỗi shipment chỉ có một mốc thời gian lấy hàng cụ thể (`pickup_date`).

Để xử lý thách thức này, chúng tôi áp dụng kỹ thuật **conditional join (non-equi join)** theo các bước:
1. Đối sánh vùng của sự kiện thời tiết với vùng gửi (`origin_region`) hoặc vùng nhận (`dest_region`) của shipment.
2. Kiểm tra điều kiện thời gian: ngày lấy hàng phải nằm trong khoảng diễn ra sự kiện (`start_date ≤ pickup_date ≤ end_date`).
3. Tổng hợp dữ liệu bằng cách lấy mức độ nghiêm trọng lớn nhất (`max_weather_severity`) và tổng số sự kiện thời tiết đồng thời (`n_weather_events`). Bước gom nhóm này giúp đảm bảo cấu trúc dữ liệu không bị nhân dòng (mỗi shipment duy trì đúng một bản ghi sau khi join).

Logic liên kết trên cũng được áp dụng tương tự tại Phần 4 để xác định loại hình thời tiết (`event_type`) có mức độ nghiêm trọng cao nhất ảnh hưởng đến đơn hàng.

## 3. Khám phá dữ liệu (EDA) - các phát hiện chính

Xếp theo mức độ liên quan đến trễ giao hàng:

1. **Độ tin cậy của đơn vị vận chuyển (Carrier)**: Đây là tín hiệu dự báo mạnh mẽ nhất. Chỉ số `reliability_score` tương quan nghịch cực kỳ chặt chẽ với tỷ lệ trễ hàng (**r = -0.957**). Trong khi hãng vận chuyển FastTrack đạt tỷ lệ trễ rất thấp (chỉ 3.8%), thì nhà xe BudgetShip (tỷ lệ trễ lên tới 27.7%) là điểm nghẽn hệ thống, xuất hiện lặp lại ở các tuyến đường có rủi ro giao trễ cao nhất như Scotland → South East (71.4%), London → Scotland (60%), và Wales → London (58%). Phân tích này khẳng định **sự chậm trễ là vấn đề nội tại của nhà xe, chứ không phụ thuộc vào tuyến đường**.
2. **Yếu tố thời tiết**: Việc lấy hàng trong giai đoạn thời tiết xấu dẫn đến tỷ lệ giao trễ lên tới **29.8%**, cao gấp gần 3.5 lần so với khi thời tiết bình thường (**8.6%**). Yếu tố này ảnh hưởng trực tiếp đến 15.5% tổng số đơn hàng.
3. **Khoảng cách tuyến đường**: Tương quan thuận ở mức trung bình (**r = 0.214**). Tỷ lệ trễ tăng dần theo cự ly vận chuyển, dao động từ 4.7% (ở chặng ngắn 0–200 km) lên đến 26.0% (ở chặng dài 600–800 km).
4. **Phân hạng dịch vụ (Service tier)**: Trực giác ban đầu thường nghĩ dịch vụ cao cấp sẽ an toàn hơn, nhưng thực tế nhóm giao hàng nhanh (express) có tỷ lệ trễ cao nhất (**16.6%**), so với nhóm tiết kiệm (economy - **9.6%**). Điều này hợp lý vì dịch vụ nhanh có khung giờ giao hàng rất hẹp, hầu như không có thời gian dự phòng để bù đắp cho các sự cố gián đoạn phát sinh.
5. **Tính chất mùa vụ**: Tỷ lệ giao hàng trễ vào các tháng cao điểm mùa đông (Tháng 12 và Tháng 1) vượt trội đáng kể so với mức trung bình năm (~11.9%). Ví dụ, tỷ lệ trễ trong Tháng 12/2025 đạt 21.5% và Tháng 1/2025 đạt 19.0%. Điều này gợi ý việc tạo thêm đặc trưng `is_winter`.
6. **Đặc trưng trọng lượng (`weight_kg`)**: Phân phối dữ liệu lệch phải rõ rệt (**skew = 2.07**) với giá trị trung bình đạt 5.06 kg, trong khi trung vị chỉ ở mức 3.45 kg.

## 4. Feature engineering

Từ bảng join gốc, xây dựng thêm **11 feature mới**, mỗi feature bám theo một phát hiện cụ thể từ EDA (không chọn ngẫu nhiên):

| Feature | Ý nghĩa |
|---|---|
| `weather_affected` | Có sự kiện thời tiết ở origin HOẶC dest |
| `weather_both_ends` | Cả origin VÀ dest cùng bị ảnh hưởng thời tiết - rủi ro cộng dồn |
| `is_winter` | Pickup rơi vào tháng 11–2, bám theo tính mùa vụ ở EDA |
| `volume_to_weight` | Tỷ trọng volume/weight - proxy cho hàng cồng kềnh/bất thường |
| `carrier_route_risk` | `(1 - reliability_score) × historical_delay_rate` - carrier kém + tuyến rủi ro cao |
| `day_of_week` / `is_monday_or_friday` | Chu kỳ workload đầu/cuối tuần ở depot |
| `is_cross_region` | Shipment liên vùng hay nội vùng |
| `n_weather_events` | Tổng số sự kiện thời tiết chồng lên ngày pickup (origin+dest) |
| `implied_speed_required` | `distance_km / số ngày cam kết theo tier` - áp lực thời gian trên quãng đường |
| `weekend_buffer_risk` | Express + pickup thứ Sáu → deadline rơi vào thứ Bảy, thời điểm depot thiếu nhân sự (delay rate 16.1% vs 11.7%) |
| `dominant_weather_event_type` | Loại thời tiết (`snow`/`flood`/`fog`/`storm`/`high_wind`/`none`) của sự kiện severity cao nhất ở origin hoặc dest - `severity` (số) coi mọi loại thời tiết cùng mức độ là như nhau, nhưng một trận lũ (`flood`) thực tế có thể chặn đường hoàn toàn theo cách mà bão gió (`storm`) cùng severity không làm được |

*(Chỉ ở `experiments.ipynb`)* Một feature thứ 12 - `n_shipments_same_origin_day` (số shipment cùng `origin_region` + `pickup_date`, proxy tắc nghẽn depot, tương tự đếm số tàu cùng cập một cảng một lúc) - được thử nhưng cho kết quả **âm tính rõ ràng**: tương quan chuẩn với `is_delayed` chỉ -0.008, và không lọt top-15 hệ số ở cả bộ full lẫn selected feature. Lý do nhiều khả năng: với ~10,025 shipment trải trên 8 vùng và 900 ngày, khối lượng cùng-vùng-cùng-ngày quá nhỏ (trung bình ~2.3, tối đa 9) để tạo ra tắc nghẽn thật ở quy mô dữ liệu này. Feature này **không được đưa vào `coursework.ipynb`** vì không mang lại giá trị.

**Quy trình lọc đặc trưng (Feature Selection)**: Sử dụng hệ số tương quan Pearson đối với các đặc trưng số (ngưỡng $|r| > 0.85$) và chỉ số Cramér's V đối với các đặc trưng phân loại nhằm phát hiện các biến bị trùng lặp thông tin. Tiêu chí loại bỏ được cân nhắc kỹ lưỡng: chỉ lược bỏ đặc trưng khi thực sự là bản sao hoặc biểu diễn lại cùng một thông tin với đặc trưng khác; giữ lại những biến phản ánh các khía cạnh vận hành riêng biệt. Ví dụ, đặc trưng `dominant_weather_event_type` được giữ lại sau khi kiểm tra qua chỉ số Cramér's V. Kết quả sau quá trình chọn lọc: loại bỏ 8 đặc trưng trùng lặp (`capacity_utilisation`, `avg_transit_days`, `during_weather`, `weather_affected`, `n_weather_events_origin`, `n_weather_events_dest`, `is_monday_or_friday`, `is_cross_region`), giữ lại **24 đặc trưng tối ưu** từ tổng số 32 biến ứng viên ban đầu.

## 5. Huấn luyện mô hình

### 5.1 Chia train/validation - điểm mấu chốt về phương pháp

Dữ liệu trải dài 900 ngày, nên **không dùng random/stratified split**: một split ngẫu nhiên khiến train và validation phủ cùng một khoảng thời gian, vô tình để mô hình "nhìn thấy tương lai" và chỉ kiểm tra khả năng nội suy chứ không phải khả năng dự báo tiến (forward forecast) - đúng như SwiftRoute cần trong production. Thay vào đó, dữ liệu được sắp theo `pickup_date`, **train trên 80% sớm nhất (8,020 shipment), validate trên 20% gần nhất (2,005 shipment)**. Tương tự, CV dùng `TimeSeriesSplit` (expanding window) thay vì `StratifiedKFold`, để không fold nào validate trên dữ liệu sớm hơn phần train của chính nó.

### 5.2 Hai model bắt buộc (+ CatBoost ở `experiments.ipynb`)

| Model | Cách xử lý imbalance | Regularisation/Tuning |
|---|---|---|
| **Logistic Regression** | `class_weight='balanced'` | L2 (ridge), `C=1.0` - phù hợp vì nhiều feature thời tiết tương quan cao với nhau, L2 co cụm hệ số thay vì zero-out như L1 |
| **LightGBM** | `class_weight='balanced'` | `RandomizedSearchCV` (30 lượt) trên `TimeSeriesSplit`, hội tụ về cây nông, learning rate thấp - hợp lý với ~10k dòng dữ liệu |

*(Chỉ trong `experiments.ipynb`)* **CatBoost** được thêm làm model thứ 3 ngoài yêu cầu, tune tương tự LightGBM, để kiểm tra xem một implementation gradient-boosting khác có đổi kết quả không.

Các model được huấn luyện trên cả bộ feature đầy đủ và bộ feature đã chọn lọc (Section 4) → 4 tổ hợp kết quả (6 tổ hợp ở `experiments.ipynb` khi tính cả CatBoost).

### 5.3 Kết quả so sánh & mô hình được chọn

| Model (feature set) | Val ROC-AUC |
|---|---|
| Logistic Regression (full features) | 0.8501 |
| Logistic Regression (selected features) | 0.8497 |
| LightGBM (full features) | 0.8447 |
| LightGBM (selected features) | 0.8439 |

*(`experiments.ipynb`: CatBoost đạt 0.8459 full-feature / 0.8468 selected-feature - nằm giữa hai mốc trên, không đổi kết luận.)*

Cả 4 tổ hợp nằm trong khoảng AUC 0.844–0.850 - sát nhau, với một overfit gap thật (~0.02 CV-vs-val) hiện rõ nhờ time-based split. Chênh lệch giữa LR full-feature và LR selected-feature chỉ ~0.0004 - nằm trong nhiễu - nên **mô hình được chọn triển khai là Logistic Regression trên bộ feature đã chọn lọc**: hệ số ổn định hơn, tránh được lệch dấu do đa cộng tuyến. Ví dụ cụ thể: ở bộ full-feature, `reliability_score` mang hệ số **+1.27** - sai dấu, ngụ ý carrier tin cậy hơn lại tăng rủi ro, trái với r=-0.957 tìm thấy ở EDA - vì gần như đối xứng với `capacity_utilisation` (r=-0.97 giữa hai cột). Sau khi bỏ `capacity_utilisation` ở bộ feature đã chọn lọc, hệ số này đảo về **-1.13**, đúng chiều - đúng là loại đa cộng tuyến mà bước chọn lọc feature muốn tránh.

### 5.4 *(Chỉ ở `experiments.ipynb`)* Thử nghiệm hệ thống: tune trực tiếp theo Recall@Top-15%

Cả 3 model (LR, LightGBM, CatBoost) còn được tune lại một lần nữa, lần này chấm điểm trực tiếp bằng **Recall@Top-15%** thay vì ROC-AUC. Đây là lần chạy **thứ ba** của thử nghiệm này (sau bản chỉ có `dominant_weather_event_type`, rồi bản thêm 3 cột carrier, giờ thêm cả `n_shipments_same_origin_day`), và mỗi lần cho một kiểu kết quả khác nhau:
- **Logistic Regression**: lần này tìm ra `C=0.03` (khác hẳn mặc định `C=1.0`) và **thực sự cải thiện** - Recall@15% từ 56.6% lên 57.1%, AUC cũng nhỉnh hơn (0.8511 vs 0.8494).
- **LightGBM**: recall-tuning **làm tệ hơn** - chỉ đạt 54.4% so với 56.6% nếu không tune theo recall. Đây là lần **thứ ba liên tiếp** (qua 3 bộ feature khác nhau) mà recall-tuning làm LightGBM tệ đi - phát hiện nhất quán nhất trong toàn bộ thử nghiệm này.
- **CatBoost**: lần này recall-tuning **làm tệ hơn** (57.5% vs 58.0% nếu để AUC-tuned) - đảo chiều so với lần chạy trước đó (từng giúp cải thiện).
- Model tốt nhất theo Recall@15% hiện tại: **CatBoost (AUC-tuned, selected features) ở 58.0%** - không phải model nào được tune riêng cho recall cả.

Bài học: việc đổi metric chấm điểm bên trong một search có ngân sách cố định không đảm bảo cải thiện - kết quả phụ thuộc nhiều vào model family và thay đổi giữa các lần chạy khi feature set thay đổi. Duy nhất phát hiện về LightGBM (recall-tuning luôn làm nó tệ hơn) là nhất quán qua cả 3 lần chạy; còn lại phần lớn nằm trong nhiễu.

## 6. Đánh giá mô hình

- **ROC curve**: các tổ hợp model/feature-set đều nằm rõ trên đường chéo random-guess, AUC 0.844–0.850 trên validation set (time-based) - thấp hơn ~0.86–0.87 từng thấy với split ngẫu nhiên trước đây, nhưng là con số **đáng tin hơn** vì phản ánh đúng khả năng dự báo tương lai.
- **Phân phối điểm dự đoán**: nhóm `is_delayed=1` lệch rõ về bên phải so với nhóm `is_delayed=0`, có overlap thật (hợp lý ở mức AUC ~0.85) - phần lớn delay thật nằm trên ngưỡng top-15%, đây chính là lý do capture rate ở Section business impact cao.
- **Hệ số Logistic Regression** (top ảnh hưởng, model đã triển khai): `origin_weather_severity` (+1.68), `dest_weather_severity` (+1.36), `reliability_score` (-1.13, đúng chiều), `service_tier_express` (+0.62), `is_winter` (+0.57), `dominant_weather_event_type_high_wind` (-0.35 - một sự kiện gió mạnh gắn với rủi ro *thấp hơn* một trận lũ cùng mức severity, hợp lý vì lũ có thể chặn đường hoàn toàn), `weekend_buffer_risk` (+0.29), `weather_both_ends` (+0.23). Đáng chú ý: **3 cột carrier mới** (`avg_delay_mins`, `tier_coverage`, `n_active_routes`) **không lọt top-15** - chúng qua được bước lọc redundancy nhưng không mang thêm tín hiệu biên đáng kể ngoài những gì `reliability_score`/`historical_delay_rate` đã nắm bắt. Một ngoại lệ: `max_weather_severity` mang dấu âm (-1.20) dù về logic phải dương - **artefact đa cộng tuyến dư** (trùng lặp thông tin với 2 cột severity origin/dest), không ảnh hưởng AUC nhưng hệ số này không nên đọc theo nghĩa đen.

## 7. Tác động kinh doanh (Business Impact)

Trên tập validation (2,005 shipment gần nhất, 226 delay thật), chính sách **flag top 15% theo rủi ro dự đoán** (301 shipment) bắt được:

- **Model policy: 127/226 delay - capture rate 56.2%** - tiết kiệm **£5,715** (giả định £45/delay tránh được)
- **Random 15% policy** (đối chứng, theo đúng yêu cầu đề bài): chỉ **33/226 (14.6%)**, tiết kiệm **£1,485**
- **Lift: 3.85x**, tương đương **£4,230 giá trị tăng thêm** so với chọn ngẫu nhiên, trên slice validation này

**Ma trận nhầm lẫn tại đúng ngưỡng top-15%** (không phải ngưỡng 0.5 mặc định):

| | Không flag | Flag (top 15%) |
|---|---|---|
| **Không trễ** | TN = 1,605 | FP = 174 |
| **Trễ** | FN = 99 | TP = 127 |

→ Precision = 42.2%, Recall = 56.2%, Macro F1 = 0.702. Precision dưới 50% có nghĩa hơn một nửa shipment bị flag hóa ra vẫn đúng hẹn - nhưng đây là hệ quả tất yếu của trần can thiệp 15%: mô hình buộc phải flag đúng 301 shipment bất kể có bao nhiêu shipment thực sự rủi ro cao.

**Khuyến nghị**: thay bất kỳ cách chọn thủ công/ngẫu nhiên nào bằng risk ranking của mô hình. Với cùng ngân sách 15%, Operations bắt được khoảng 5–6/10 delay thật thay vì ~1.5/10 - mô hình không tốn thêm chi phí vận hành, chỉ giúp nhắm đúng năng lực can thiệp sẵn có vào đúng shipment cần nó.

### 7.1 Độ nhạy Precision/Recall theo ngân sách can thiệp

Ngân sách 15% là ràng buộc cố định từ đề bài, nhưng để hiểu rõ hơn vị trí của nó trên đường cong đánh đổi precision/recall, cùng một model (không train lại) được thử ở 10% và 20%:

| Ngân sách | Flagged | Bắt được delay | Recall | Precision | F1 | Tiết kiệm |
|---|---|---|---|---|---|---|
| **10%** | 201 | 102 | 45.1% | 50.7% | 0.478 | £4,590 |
| **15%** (hiện tại) | 301 | 127 | 56.2% | 42.2% | 0.482 | £5,715 |
| **20%** | 401 | 149 | 65.9% | 37.2% | 0.475 | £6,705 |

Đánh đổi kinh điển: ở 10%, precision đạt ~50% (gần một nửa shipment bị flag đúng là sẽ trễ); ở 20%, recall tăng lên 65.9% nhưng precision giảm còn 37.2% (gần 2/3 số can thiệp là báo động giả). F1 gần như phẳng (0.478 → 0.482 → 0.475) — không có mức nào áp đảo rõ ràng trên một metric tổng hợp. Số £ tiết kiệm tăng đơn điệu theo ngân sách nhưng dễ gây hiểu lầm: công thức chỉ tính lợi ích khi bắt đúng, không trừ chi phí can thiệp vào các shipment hóa ra vẫn đúng hẹn. Vì 15% là ràng buộc cứng chứ không phải tham số tự do, phân tích này chỉ mang tính đối chiếu — nhưng xác nhận 15% nằm ở vị trí hợp lý giữa hai thái cực.

*(`experiments.ipynb` cho kết quả tương tự về hình dạng đánh đổi, với baseline khác một chút — 10%: 99/226 (43.8%), 15%: 125/226 (55.3%), 20%: 150/226 (66.4%) — do bộ feature ở đó có thêm `route_risk_cluster`.)*

## 8. *(Chỉ ở `experiments.ipynb`)* Thử nghiệm mở rộng: Blend CatBoost + Logistic Regression

Recall 56.6% (baseline experiments.ipynb) vẫn để lọt gần một nửa số delay thật. Thử blend trung bình có trọng số giữa xác suất dự đoán của LR và CatBoost (recall-tuned), trọng số tune qua CV theo `TimeSeriesSplit`. Kết quả lần này: trọng số tối ưu nghiêng hẳn về CatBoost (`w=0.10`, tức chỉ 10% LR / 90% CatBoost), đạt CV Recall@15% đỉnh 62.1%. Trên validation set thật, blend ở trọng số này cho **130/226 (57.5%)** - cải thiện thật so với baseline (128/226, 56.6%, +£90) và so với LR recall-tuned ở 8.3 (129/226, 57.1%, +£45) - nhưng **trùng khớp tuyệt đối với CatBoost (recall-tuned) một mình**, vì một blend nghiêng 90% về CatBoost gần như không khác gì CatBoost đứng riêng.

**Đánh giá thực tế**: Hiệu quả cải thiện từ phương pháp blend thực chất chỉ phản ánh hiệu năng của mô hình CatBoost (do tỷ trọng đóng góp lên tới 90%), chứ không đến từ sự cộng hưởng giá trị giữa hai thuật toán khác biệt. Khi kết hợp với kết quả tại Mục 5.4 (mô hình CatBoost tối ưu theo AUC đạt hiệu năng Recall@15% tốt nhất ở mức 58.0%), xu hướng chung qua các thử nghiệm cho thấy: các mô hình dạng cây (tree-based) có xu hướng nhỉnh hơn Logistic Regression về chỉ số Recall@Top-15%, tuy nhiên kết quả này dao động tùy thuộc vào bộ đặc trưng. Mặc dù vậy, Logistic Regression vẫn là lựa chọn ưu tiên để triển khai thực tế nhờ vào khả năng giải thích mô hình (interpretability) vượt trội (Mục 6.4).

## 9. Giám sát drift khi lên production

- **Input drift**: so sánh phân phối từng feature theo cửa sổ trượt (tuần) với phân phối lúc train - PSI hoặc Kolmogorov-Smirnov cho numeric (`weight_kg`, `distance_km`, `max_weather_severity`), chi-square cho categorical (`service_tier`, `carrier_id`). Nhóm dễ drift nhất: mix `service_tier`/`carrier_id` (do quyết định thương mại của SwiftRoute) và các feature thời tiết (mùa vụ tự nhiên). Ngưỡng cảnh báo: PSI > 0.2 hoặc p-value có ý nghĩa thống kê **và** effect size đủ lớn (tránh báo động giả do cỡ mẫu lớn).
- **Target drift**: theo dõi hàng tuần (1) tỷ lệ trễ thực tế so với baseline ~11.9%, (2) ROC-AUC/calibration thực tế so với baseline ~0.85 (đo được vài ngày sau khi có kết quả giao hàng thật). Kích hoạt retrain khi: AUC live giảm rõ rệt và kéo dài nhiều tuần, capture rate ở ngưỡng 15% giảm, hoặc có thay đổi cấu trúc mới (carrier mới, vùng mới) mà mô hình chưa từng thấy.

## 10. Extension

- **E1 - Phân cụm tuyến đường**: KMeans (3 cụm) trên `distance_km`, `n_depot_hops`, `historical_delay_rate`, `avg_transit_days` cho ra một cụm "long-haul" rõ rệt (22 tuyến, TB 532km, 3 hop, **24.4%** delay rate lịch sử) so với cụm "short-haul" (16 tuyến, TB 161km, 1 hop, **10.4%**) - rủi ro hơn gấp 2 lần. Cụm long-haul tập trung quanh London, Wales, South_East, Scotland - khớp với các tuyến/carrier rủi ro nhất đã thấy ở EDA. Kết luận: rủi ro của mạng lưới SwiftRoute mang tính **cấu trúc** (tuyến dài + nhiều hop), tập trung ở vài hành lang cụ thể chứ không rải đều.
- **E2 - SHAP**: giải thích 10 shipment rủi ro cao nhất (predicted risk 98.7–99.9%) - `origin_weather_severity`/`dest_weather_severity` luôn là 2 yếu tố đóng góp lớn nhất, tiếp theo là `reliability_score` thấp. Đây là các trường hợp thời tiết xấu + carrier yếu cộng dồn. `max_weather_severity` vẫn cho SHAP âm dù shipment rủi ro cao - cùng artefact đa cộng tuyến đã nêu ở Section 6.

## 11. Hướng đã cân nhắc nhưng không áp dụng

**SMOTE (Synthetic Minority Over-sampling Technique)**: Phương pháp này đã được cân nhắc nhằm cải thiện chỉ số Recall nhưng không được áp dụng vì ba nguyên nhân chính:
1. Việc thiết lập tham số `class_weight='balanced'` đã giải quyết hiệu quả vấn đề mất cân bằng lớp (class imbalance) trong quá trình huấn luyện; kết hợp thêm SMOTE có thể dẫn đến hiện tượng hiệu chỉnh quá mức (over-correction) mà không mang lại hiệu quả cộng dồn.
2. Chỉ số đánh giá cốt lõi là Recall tại **ngưỡng top-15% phân hạng xác suất**, không phải tại ngưỡng mặc định 0.5. SMOTE chủ yếu dịch chuyển ngưỡng quyết định (decision boundary) thay vì làm thay đổi thứ tự xếp hạng tương đối giữa các đơn hàng.
3. Rủi ro kỹ thuật phát sinh khi nội suy SMOTE trên các biến nhị phân hoặc biến phân loại đã mã hóa một nóng (one-hot), dễ tạo ra các mẫu dữ liệu giả lập không thực tế (trừ khi sử dụng SMOTENC). Đồng thời, việc phải áp dụng SMOTE trong từng phân đoạn của `TimeSeriesSplit` để tránh rò rỉ dữ liệu (data leakage) làm tăng đáng kể độ phức tạp của hệ thống mà không đảm bảo hiệu quả rõ rệt.

## 12. Tổng kết

- Mô hình triển khai: **Logistic Regression (bộ feature đã chọn lọc, 24 feature)**, Val ROC-AUC ≈ 0.850, lưu tại `model.pkl`.
- Tại ngưỡng vận hành thực tế (top 15%): capture rate 56.2% (127/226) lift 3.85x so với chọn ngẫu nhiên.
