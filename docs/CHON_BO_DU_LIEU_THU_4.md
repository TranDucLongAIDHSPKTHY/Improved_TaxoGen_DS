# Chọn bộ dữ liệu thứ tư (có cây danh mục chuẩn) — quy tắc chốt trước

Ngày chốt: 2026-10-28. Văn bản này và script `tools/survey_amazon_categories.py` được
commit **trước khi** chạy khảo sát; mọi con số khảo sát chỉ được xem sau commit này.

## 1. Câu hỏi cần trả lời

Trên Handmade Products (bộ duy nhất có cây chuẩn), Improved khớp cây danh mục Amazon
kém hơn TaxoGen (pairF1@1, NMI@1) và thấp hơn về NPMI. Câu hỏi: đây là điểm yếu chung
khi dựng cây từ review sản phẩm, hay đặc thù của Handmade? Bộ thứ tư phải cùng loại dữ
liệu (review + cây danh mục Amazon Reviews 2023) nhưng khác lĩnh vực.

## 2. Đã biết trước khi chốt (không phải kết quả của phương pháp nào)

- Kích thước file review đã chạy: Amazon Fashion 292.425.439 byte, Handmade Products
  77.803.288 byte.
- Khi chọn Handmade (2026-10-26) đã xem kích thước file và một mẫu đường danh mục của một số
  danh mục. Chưa chạy phương pháp nào (dựng cây, gợi ý) trên bất kỳ danh mục ứng viên nào.

## 3. Quy tắc (áp dụng máy móc, theo thứ tự)

**D — lĩnh vực.** Chỉ xét danh mục sản phẩm vật lý, review nói về chính sản phẩm. Loại:
- đã dùng: Amazon_Fashion, Handmade_Products;
- trùng lĩnh vực với bộ đã dùng: Clothing_Shoes_and_Jewelry (Fashion), Arts_Crafts_and_Sewing (Handmade);
- nội dung/số hoá hoặc không phải sản phẩm: Books, Kindle_Store, Movies_and_TV, CDs_and_Vinyl,
  Digital_Music, Video_Games, Software, Magazine_Subscriptions, Gift_Cards, Subscription_Boxes, Unknown.

**S — cỡ dữ liệu** (Content-Length của file review, đọc bằng HTTP HEAD, không tải):
Handmade (77.803.288) ≤ cỡ ≤ Fashion (292.425.439). Cận trên: chuỗi chạy đã chứng minh vừa
máy và thời gian; cận dưới: tránh lặp lại vấn đề cây quá nhỏ của Handmade (train split không
đạt 50 nút), để thất bại ở Handmade không bị giải thích bằng cỡ corpus.

**G — chất lượng cây chuẩn** (đọc toàn bộ file metadata, chỉ trường `categories`; bỏ tầng
đầu là tên phòng ban, đúng như `evaluate-taxonomy` làm):
- G1: ≥ 80% sản phẩm có ít nhất 1 tầng sau khi bỏ tầng đầu (dùng được để chấm);
- G2: trong số đó, ≥ 80% có ít nhất 2 tầng (chấm được độ sâu 1 và 2);
- G3: số nhánh cấp 1 chiếm mỗi nhánh ≥ 1% sản phẩm nằm trong [5, 50];
- G4: nhánh cấp 1 lớn nhất chiếm ≤ 50% sản phẩm (tránh phép chia gần như tầm thường).

**T — chọn một.** Trong các danh mục qua D, S, G: chọn danh mục có file review **lớn nhất**
(nhiều dữ liệu nhất trong giới hạn; tách rõ nhất khỏi giải thích "Handmade nhỏ").

Nếu không danh mục nào qua: **không** thêm bộ thứ tư; không nới quy tắc.

## 4. Cam kết khi chạy

- Config sao chép `configs/handmade_products.yaml`, chỉ đổi tên danh mục và đường dẫn:
  cùng lưới tham số, `min_nodes: 30`, `gold: true`, `depths: [1, 2, 3]`, không chấm người,
  cùng phần gợi ý. Không sửa code phương pháp.
- Chọn tham số trên val; mở test một lần (`eval/test_access.json`).
- Mọi kết quả (thắng, thua, hoà) được báo cáo trong bài hoặc phụ lục; bộ dữ liệu không bị
  bỏ sau khi thấy kết quả.
- Bảng khảo sát đầy đủ (mọi danh mục, kể cả bị loại) được lưu ở
  `outputs/dataset_survey/` và báo cáo kèm.

## 5. Kết quả khảo sát (2026-10-28, sau commit 221a08f)

Bảng đầy đủ: `outputs/dataset_survey/survey_20261028-080610.json`. Chỉ hai danh mục qua D và S:

| danh mục | file review | G1 | G2 | G3 | G4 (nhánh lớn nhất) | kết quả |
|---|---|---|---|---|---|---|
| All_Beauty | 94 MB | 0.000 | — | 0 | — | không có đường danh mục |
| Appliances | 270 MB | 0.965 | 0.802 | 5 | 0.783 (Parts & Accessories) | trượt G4 |

Các danh mục sản phẩm còn lại trượt S (Health_and_Personal_Care 70 MB, nhỏ hơn Handmade;
mọi danh mục khác lớn hơn Fashion, nhỏ nhất là Musical_Instruments 459 MB).

**Kết luận theo quy tắc: không thêm bộ dữ liệu thứ tư.** Quy tắc không được nới sau khi đã
thấy bảng này.
