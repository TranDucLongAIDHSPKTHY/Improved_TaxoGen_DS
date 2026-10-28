# Chọn hai bộ dữ liệu xác nhận — quy tắc chốt trước

Ngày chốt: 2026-10-28. Văn bản này được commit **trước khi** tải hay mở bất kỳ dữ liệu nào của
các ứng viên dưới đây. Đi kèm `docs/BAN_CONG_BO.md` (mục 1): có đúng **hai** bộ xác nhận, một bộ
thương mại điện tử (đối chứng với Fashion và Handmade) và một bộ văn bản khoa học (đối chứng với
DBLP). Cả hai đều được chấm bởi người ở vòng 2 nên không cần cây danh mục chuẩn.

## 1. Đã biết trước khi chốt (không phải kết quả của phương pháp nào)

- Kích thước file review của mọi danh mục Amazon Reviews 2023 (HTTP HEAD) và thống kê đường danh
  mục của All_Beauty và Appliances, từ khảo sát cho bộ thứ tư
  (`docs/CHON_BO_DU_LIEU_THU_4.md`, `outputs/dataset_survey/survey_20261028-080610.json`). Khảo
  sát đó trả lời câu hỏi khác (cây chuẩn) và kết luận của nó giữ nguyên; quy tắc dưới đây không
  dùng cây chuẩn.
- Chưa chạy phương pháp nào (dựng cây, gợi ý) trên bất kỳ ứng viên nào. Chưa tải dữ liệu arXiv
  hay ACL Anthology.

## 2. Bộ thương mại điện tử

**D — lĩnh vực** (giữ nguyên quy tắc D của khảo sát trước): danh mục sản phẩm vật lý của Amazon
Reviews 2023; loại Amazon_Fashion, Handmade_Products (đã dùng), Clothing_Shoes_and_Jewelry,
Arts_Crafts_and_Sewing (trùng lĩnh vực), Books, Kindle_Store, Movies_and_TV, CDs_and_Vinyl,
Digital_Music, Video_Games, Software, Magazine_Subscriptions, Gift_Cards, Subscription_Boxes,
Unknown (nội dung số hoá hoặc không phải sản phẩm).

**S — cỡ**: Fashion (292.425.439 byte) ≤ file review ≤ 1.000.000.000 byte. Cận dưới: cỡ nhỏ
nhất đã chứng minh cây đạt 50 nút trên tập train (Handmade, 78 MB, không đạt); cận trên: tải và
chạy toàn bộ chuỗi (gồm gợi ý) vừa máy 32 GB trong vài ngày.

**T — chọn**: sắp các danh mục qua D và S theo tên (thứ tự chữ cái), chọn một bằng
`numpy.random.default_rng(20261028).choice(danh_sách)`. Rút thăm thay cho tiêu chí nội dung để
không ai chọn được danh mục theo dự đoán kết quả. Chạy bằng
`tools/draw_confirmation_category.py` sau khi commit văn bản này; kết quả rút thăm là cuối cùng.

**Chạy**: config sao chép `configs/amazon_fashion.yaml`, chỉ đổi danh mục và đường dẫn; nếu file
review lớn hơn 1,5 lần Fashion, corpus taxonomy dùng `corpus.sample_fraction` =
292.425.439 / cỡ file (mẫu tất định có sẵn), phần gợi ý dùng toàn bộ tương tác.

## 3. Bộ văn bản khoa học

Người chấm là sinh viên, giảng viên CNTT, nên bộ khoa học phải thuộc khoa học máy tính; khác
DBLP về nguồn và loại văn bản (tóm tắt thay vì tiêu đề).

**Ứng viên theo thứ tự ưu tiên** (lấy ứng viên đầu tiên khả thi; khả thi = tải được không cần
đăng nhập và đọc được đúng định dạng mô tả, kiểm tra trên vài bản ghi, không chạy phương pháp):

1. **arXiv, khoa học máy tính**: thu thập qua OAI-PMH của arXiv (`set=cs`,
   `metadataPrefix=arXiv`); giữ bài có **danh mục chính** (danh mục đầu tiên) bắt đầu bằng `cs.`
   và ngày tạo phiên bản 1 trong 2015-01-01 … 2026-09-30; văn bản = tiêu đề + tóm tắt. Siêu dữ
   liệu arXiv theo giấy phép CC0.
2. **ACL Anthology**: file `anthology+abstracts.bib.gz` tại ngày tải; bài có tóm tắt; văn bản
   = tiêu đề + tóm tắt.

**Cỡ**: nếu nhiều hơn 500.000 tài liệu, giữ mẫu tất định 500.000 tài liệu theo băm
(`crc32("20261028:" + id)`, lấy nhỏ nhất). Lý do: tóm tắt dài hơn tiêu đề DBLP nhiều lần;
500.000 tóm tắt vẫn lớn hơn Fashion về số từ.

**Chạy**: tách từ và trích cụm từ bằng cùng pipeline của Amazon (không có danh sách cụm từ
AutoPhrase như DBLP); cùng lưới tham số, quy tắc chọn, `min_nodes` 50; không có phần gợi ý.
File thu thập được ghi sha256 trước khi tách từ.

## 4. Cam kết

- Hai bộ được chọn là cuối cùng: không bỏ, không thay, không rút thăm lại sau khi xem dữ liệu
  hay kết quả. Nếu một bộ không chạy được về kỹ thuật (ví dụ không cấu hình nào đạt 50 nút trên
  train), điều đó được báo cáo như một kết quả; việc thay thế (nếu có) phải được ghi vào mục 5
  trước khi xem bất kỳ chỉ số nào của bộ thay thế, và bộ ban đầu vẫn được báo cáo.
- Code cần thêm để đọc dữ liệu (bộ đọc OAI-PMH / bib, config) chỉ đọc và chuyển định dạng; không
  đổi phương pháp. Mọi kết quả được báo cáo theo `docs/BAN_CONG_BO.md` mục 8.

## 5. Kết quả chọn và thay đổi

- 2026-10-28, sau commit 2de4f0c: `tools/draw_confirmation_category.py` cho ba danh mục qua D
  và S: Baby_Products, Industrial_and_Scientific, Musical_Instruments; rút thăm (seed 20261028):
  **Baby_Products** (file review 837.501.465 byte, lớn hơn 1,5 lần Fashion, nên
  `corpus.sample_fraction` = 292.425.439 / 837.501.465 ≈ 0.3492). Chưa tải dữ liệu.
- Bộ khoa học: chưa kiểm tra tính khả thi của arXiv OAI-PMH.
