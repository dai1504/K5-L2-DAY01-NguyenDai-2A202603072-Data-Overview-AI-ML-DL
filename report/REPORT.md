# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về **cả bức ảnh** vì đây là bài toán Image Classification. Model đưa ra một nhãn đại diện cho nội dung hình ảnh, không xác định vị trí hay tọa độ của từng vật thể riêng biệt.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp được định nghĩa sẵn và đi kèm với taxonomy/checkpoint của model. Model không tự nghĩ ra tên lớp mới mà chỉ dự đoán trong danh sách lớp đã được huấn luyện.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần lưu cả ba thông tin để xác định lớp chính xác. class_id là mã định danh của lớp trong taxonomy, class_name là tên lớp dễ đọc và taxonomy_name cho biết lớp thuộc hệ thống phân loại nào. Vì cùng một ID có thể mang ý nghĩa khác nhau trong các taxonomy khác nhau nên không nên chỉ lưu tên lớp.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ tiêu chí lựa chọn một lớp duy nhất, ví dụ ưu tiên chủ thể chính hoặc đối tượng nổi bật nhất trong ảnh. Nếu có nhiều chủ thể quan trọng hoặc không xác định được nội dung chính thì cần có quy tắc xử lý hoặc chuyển sang escalation.
6. Vì sao không được lấy `score` của model làm ground truth?
Score chỉ thể hiện mức độ tin cậy của model đối với dự đoán, không đảm bảo kết quả chính xác. Model có thể dự đoán sai do ảnh mờ, nhiều vật thể hoặc dữ liệu phức tạp. Ground truth phải được xác định dựa trên hình ảnh thực tế và guideline do con người xây dựng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chọn đối tượng `person`:
- `class_name` = person
- `score` = 0.912624
- `bbox_xyxy` = [385.33, 69.24, 498.92, 348.92]
- `bbox_width` = 113.58
- `bbox_height` = 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Box có tọa độ:
- Góc trên-trái: (385.33, 69.24).
- Góc dưới-phải: (498.92, 348.92).
- Chiều rộng: 113.58 pixel.
- Chiều cao: 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Ngưỡng 0.20: Có 17 vật thể được phát hiện (bao gồm cả các chi tiết nhỏ, mờ hoặc độ tin cậy thấp).
- Ngưỡng 0.60: Có 6 vật thể đạt chuẩn (chỉ giữ lại các đối tượng có độ tin cậy cao, loại bỏ các dự đoán yếu).
Khi ngưỡng confidence thấp hơn, model giữ lại nhiều prediction hơn. Khi tăng ngưỡng, chỉ những prediction có score cao hơn mới được giữ lại.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Reviewer phải kiểm tra nhiều bounding box hơn, phát hiện các đối tượng bị nhận diện sai, box trùng lặp, box không chính xác hoặc đối tượng không tồn tại. Sau đó cần chỉnh sửa hoặc loại bỏ các nhãn sai để đảm bảo chất lượng dữ liệu.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Vẽ bounding box sát với đường biên ngoài của vật thể nhìn thấy được, không lấy quá nhiều phần nền. Với vật thể bị che khuất, thực hiện theo quy tắc occlusion trong guideline; không tự suy đoán phần không nhìn thấy.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong ảnh kitchen, các vật dụng ở phía bên trái bị cắt một phần bởi mép ảnh. Tôi sẽ đưa trường hợp này thành quy tắc trong guideline: chỉ gán nhãn phần vật thể còn nhìn thấy trong ảnh, vẽ bounding box hoặc polygon sát theo phần nhìn thấy và không tự suy đoán phần nằm ngoài khung hình. Đây là trường hợp có thể gặp thường xuyên nên cần có quy tắc thống nhất để tất cả annotator thực hiện giống nhau. Không cần escalation nếu guideline đã quy định rõ cách xử lý.
Vì sao câu này tốt hơn?
- Chỉ rõ một vật thể thực tế trong ảnh.
- Xác định đúng tình huống: bị cắt mép ảnh (truncated object).
- Đưa ra cách xử lý cụ thể.
- Giải thích được tại sao chọn guideline thay vì escalation.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-006" ; `class_name`="bowl" ; `score`=0.528203 ; số điểm polygon=`polygon_point_count`=84 ; 3–5 điểm đầu của `polygon_xy`=[
        29.0,
        303.0
      ],
      [
        28.0,
        304.0
      ],
      [
        26.0,
        304.0
      ],
      [
        25.0,
        305.0
      ],
      [
        23.0,
        305.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
box chỉ xác định vùng hình chữ nhật bao quanh vật thể. Polygon mô tả đường biên chi tiết hơn, giúp xác định hình dạng thực tế của vật thể và phân biệt phần vật thể với phần nền xung quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
`instance_id` dùng để định danh một đối tượng cụ thể trong ảnh, không phải class ID vì nhiều đối tượng khác nhau có thể cùng một class, không phải tracking ID vì nó không dùng để theo dõi cùng một đối tượng qua nhiều khung hình video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ polygon sát theo đường viền của vật thể, bao gồm toàn bộ phần nhìn thấy được và không lấy phần nền bên ngoài. Khi vật thể bị che khuất, chỉ vẽ phần nhìn thấy được, không tự suy đoán phần bị che.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Ở khu vực các chiếc bát trên bàn có nhiều vật thể nằm gần và chạm nhau, khiến việc xác định đường biên giữa các vật thể có thể khó khăn. Tôi sẽ áp dụng guideline nếu đã có quy tắc phân biệt biên giữa các vật thể. Nếu đường biên không thể xác định rõ ràng, tôi sẽ escalation để hỏi Lab Coach nhằm tránh gán nhãn sai.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn đơn/đa nhãn (ví dụ: traffic, kitchen) gán cho toàn bộ khung hình. | Model nhầm lẫn ảnh giao thông thành cab (score 0.5109) hoặc ảnh bếp thành gong do đặc trưng nền tương đồng. | Quan sát toàn cảnh, gán nhãn chính xác theo taxonomy. | Kiểm tra nhãn khớp với chủ thể chính, duyệt hoặc gửi rework. |
| Phát hiện vật thể | Tập hợp các bounding box có tọa độ [xmin, ymin, xmax, ymax] và ID. | Vật thể chồng chéo hoặc bị cắt cụt ở biên (ví dụ: bus score 0.9125, car score 0.8630). | Vẽ hộp bao khít ranh giới thực tế của từng đối tượng. | Kiểm tra độ khít của bounding box, đảm bảo không bỏ sót vật thể. |
| Instance segmentation | Đường viền đa giác (Polygon) hoặc mask pixel bám sát từng thực thể. | Biên vật thể phức tạp, chi tiết mỏng hoặc dính liền nhau (ví dụ: traffic-001 - bus, score 0.9257). | Dùng công cụ vẽ đa giác bám sát chi tiết biên, tránh thủng lỗ/đứt đoạn. | Phóng to kiểm tra kỹ các vùng biên phức tạp và lỗi dính liền thực thể. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng dữ liệu được cung cấp cho bài thực hành, không chia sẻ dữ liệu cá nhân hoặc dữ liệu nhạy cảm ra bên ngoài và không tự ý chỉnh sửa hay phát tán dữ liệu khi chưa được phép.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng xử lý dữ liệu đó và báo cho Lab Coach hoặc người phụ trách để được hướng dẫn, không tự ý sử dụng hoặc gán nhãn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
