# ĐỀ CƯƠNG ĐÁNH GIÁ CÓ NGƯỜI THAM GIA
(đề cương nội bộ của nhóm nghiên cứu: thí nghiệm không qua hội đồng đạo đức, người tham gia
tự xác nhận đồng ý hoặc không đồng ý ở đầu biểu mẫu; điền các mục trong ngoặc vuông)

| Mục | Nội dung |
|---|---|
| Tên nghiên cứu | Đánh giá chất lượng cây phân cấp chủ đề được xây dựng tự động (TaxoGen và các phương pháp cải tiến) |
| Nghiên cứu viên chính | Trần Đức Long — tổ chức đánh giá, giữ khoá giải mã phương pháp, không tham gia chấm |
| Đồng nghiên cứu viên | [Tên các đồng tác giả] — không tham gia chấm |
| Đơn vị | Khoa Công nghệ thông tin, Trường Đại học Công nghệ Kỹ thuật Hưng Yên |
| Thời gian dự kiến | [từ ngày … đến ngày …] |

## 1. Mục tiêu
So sánh chất lượng các cây phân cấp chủ đề do hai phương pháp tự động tạo ra, qua
hai nhiệm vụ đánh giá chuẩn trong lĩnh vực: (1) độ đúng quan hệ cha–con,
(2) độ mạch lạc chủ đề (nhận diện từ lạc đề). Người chấm không biết phương pháp
nào tạo ra câu hỏi nào (đánh giá mù).

## 2. Người tham gia
- 20 người: sinh viên và giảng viên ngành Khoa học máy tính thuộc nhóm nghiên cứu, không phải tác giả và chưa xem các cây phân cấp.
- Thêm 2–3 người cho vòng chấm thử (pilot); những người này **không** tham gia vòng chính.
- Tiêu chí loại trừ: tác giả của nghiên cứu, người đã xem các cây phân cấp, người dưới 18 tuổi.
- Tuyển chọn: thông báo mở; tham gia tự nguyện, không thù lao. Giảng viên trong nhóm nghiên cứu không trực tiếp mời sinh viên mình đang giảng dạy, để tránh áp lực tham gia.

## 3. Quy trình
1. Người tham gia đọc phiếu thông tin và nhận một mã người chấm (A01–A20).
2. Hai buổi làm trực tuyến trên Google Forms, mỗi buổi khoảng 20–30 phút (khoảng 225 câu hỏi mỗi buổi; vòng làm thử: 15–20 phút).
3. Câu đầu của mỗi biểu mẫu là Đồng ý / Không đồng ý tham gia (chọn Không đồng ý thì biểu mẫu kết thúc và dòng đó không được phân tích); sau đó là ô nhập mã người chấm, các câu hỏi và một số câu kiểm tra mức độ chú ý (câu có đáp án hiển nhiên).
4. Không có can thiệp, không thu thập dữ liệu sinh trắc hay dữ liệu nhạy cảm.

## 4. Dữ liệu thu thập và xử lý
- Thu thập: mã người chấm, câu trả lời, thời điểm nộp (Google Forms tự ghi).
- Không thu thập: họ tên, email, thông tin cá nhân khác trên biểu mẫu (tắt tính năng thu email).
- Bảng đối chiếu họ tên–mã do nghiên cứu viên chính lưu riêng (ngoại tuyến), huỷ sau [thời hạn].
- File xuất từ Google Forms được lưu nguyên bản; mã băm sha256 của từng file được ghi lại để chứng minh dữ liệu không bị chỉnh sửa.
- Trước khi phân tích, một bước kiểm tra tự động phát hiện phản hồi bất thường (trả lời đồng loạt, nộp gần như cùng lúc, trượt câu kiểm tra chú ý). Phản hồi trượt câu kiểm tra bị loại theo quy tắc định trước; số lượng bị loại được báo cáo trong bài.
- Công bố: câu trả lời chỉ kèm mã người chấm được đưa lên kho lưu trữ mở (Zenodo) cùng bài báo.

## 5. Rủi ro, lợi ích, rút lui
- Rủi ro tối thiểu (chỉ tốn thời gian). Không có lợi ích trực tiếp.
- Người tham gia có thể dừng bất cứ lúc nào và yêu cầu xoá dữ liệu trước [ngày chốt dữ liệu].

## 6. Kế hoạch phân tích (chốt trước khi gửi biểu mẫu)
Kế hoạch này và bộ câu hỏi được "đóng băng" trước khi gửi biểu mẫu: mục này nằm
trong lịch sử git trước lệnh `freeze`, và `freeze` ghi mã băm của bộ câu hỏi và
phiên bản code. Mọi quy tắc dưới đây được code áp dụng tự động
(`taxorec.human_eval.analyze`), không chọn sau khi thấy kết quả.

**6.1. Chỉ số và kiểm định chính.** Với mỗi bộ dữ liệu (Amazon Fashion, DBLP) và mỗi
nhiệm vụ: độ chính xác gộp của mỗi phương pháp, tức tỉ lệ phán đoán "đúng" trên các câu
thật (quan hệ cha–con: trả lời `yes`; từ lạc đề: chọn đúng từ được cài). So sánh
Improved − TaxoGen bằng bootstrap hai chiều (5000 lần; lấy mẫu lại câu hỏi trong từng
phương pháp và lấy mẫu lại người chấm), khoảng tin cậy 95% và p hai phía. Có đúng
**4 so sánh chính** (2 bộ dữ liệu × 2 nhiệm vụ), hiệu chỉnh Holm ở mức 0.05. Improved
chỉ được coi là tốt hơn trong một so sánh khi chênh lệch dương và p sau hiệu chỉnh Holm
< 0.05; p lớn không được diễn giải là "hai phương pháp tương đương".

**6.2. Chỉ số phụ** (báo cáo, không dùng để kết luận chính): độ chính xác theo đa số
người chấm; độ chính xác theo từng cấp của cây; khoảng Wilson (mô tả); số người chấm
có độ chính xác với Improved cao hơn / thấp hơn với TaxoGen, kèm kiểm định dấu chính xác
hai phía (bỏ các trường hợp bằng nhau).

**6.3. Độ đồng thuận giữa người chấm.** Tính trên tính đúng/sai (nhị phân) của từng
phán đoán, trên các câu thật, sau khi loại người chấm theo mục 6.4: Fleiss' κ (khi mọi
câu có cùng số người chấm) và Krippendorff's α (thang định danh). Thang diễn giải cố định:

| Fleiss' κ (Landis & Koch, 1977) | Mức |
|---|---|
| < 0 | poor (kém) |
| 0.00–0.20 | slight (rất thấp) |
| 0.21–0.40 | fair (khá thấp) |
| 0.41–0.60 | moderate (trung bình) |
| 0.61–0.80 | substantial (đáng kể) |
| 0.81–1.00 | almost perfect (gần tuyệt đối) |

| Krippendorff's α (Krippendorff, 2004) | Mức |
|---|---|
| ≥ 0.800 | reliable: đủ tin cậy |
| 0.667–0.799 | tentative: chỉ cho phép kết luận dè dặt |
| < 0.667 | low: độ tin cậy thấp |

Hệ quả, chốt trước: độ đồng thuận **không** thay đổi phân tích chính và **không** là
căn cứ để loại người chấm hay câu hỏi. Nếu α < 0.800, bài báo ghi rõ mức tương ứng,
trình bày kết luận ở mục 6.1 là dè dặt và báo cáo kèm kiểm định dấu theo người chấm
(mục 6.2); nếu α < 0.667, bài báo nêu rõ rằng các phán đoán có tính chủ quan cao.

**6.4. Loại người chấm (chỉ theo các quy tắc này).**
- Câu kiểm tra mức độ chú ý: sai từ 2 câu kiểm tra trở lên trong một nhiệm vụ thì người
  đó bị loại khỏi nhiệm vụ đó (quan hệ cha–con có 4 câu kiểm tra, từ lạc đề có 3 câu);
  sai 1 câu không bị loại.
- Kiểm tra tính toàn vẹn (`integrity_report`): nếu có cờ `critical` (nhiều cặp người
  chấm trả lời giống hệt nhau, phân bố câu trả lời đồng đều bất thường, phần lớn bài nộp
  cách nhau dưới 2 giây) thì dừng phân tích, tìm nguyên nhân và báo cáo trong bài.
- Không loại người chấm hay câu hỏi vì bất kỳ lý do nào khác. Số người bị loại ở mỗi
  nhiệm vụ được báo cáo.

## 7. Tài liệu đính kèm
- Phiếu thông tin và xác nhận đồng ý (`PHIEU_THONG_TIN_VA_DONG_Y.md`)
- Hướng dẫn chấm song ngữ (sinh tự động: `INSTRUCTIONS.md`)
- Bản xem trước biểu mẫu Google Forms
