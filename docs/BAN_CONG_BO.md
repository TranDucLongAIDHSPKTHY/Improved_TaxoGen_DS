# Bản công bố — quy tắc chốt trước

Ngày chốt: 2026-10-28. Văn bản này được commit trong repo khám phá (repo hiện tại) **trước khi**
tạo repo công bố và trước mọi bước chạy của bản công bố, rồi được chép nguyên văn làm commit
đầu tiên của repo mới. Mọi thay đổi sau ngày chốt được ghi ở mục 9, có ngày và lý do, trước khi
xem kết quả bị ảnh hưởng.

## 1. Khung nghiên cứu

| vai trò | bộ dữ liệu | cách đọc |
|---|---|---|
| Phát triển | Amazon Fashion, DBLP, Handmade Products | báo cáo đầy đủ, nêu rõ đã dùng khi phát triển phương pháp |
| Xác nhận độc lập | 1–2 bộ mới, chọn theo quy tắc ở `docs/CHON_BO_XAC_NHAN.md` (chốt trước khi xem bất kỳ dữ liệu nào) | phương pháp, baseline, lưới tham số, quy tắc đều chốt trước; test đọc **một lần** |

Chỉ các bộ xác nhận được gọi là bằng chứng độc lập. Kết quả trên bộ phát triển được trình bày
là kết quả phát triển, kể cả khi thuận lợi cho Improved.

## 2. Phương pháp Improved — đóng băng

- Phương pháp là cấu hình `taxonomy.methods.improved`, lưới `taxonomy.tuning.grids.improved`
  và quy tắc chọn `select_params` (NPMI, subsumption, sibling overlap; term coverage ≥ 0.3;
  min_nodes 50) như ở commit chứa văn bản này. Tham số được chọn lại trên val của từng bộ bằng
  đúng quy tắc đó; không thêm, bớt hay đổi giá trị nào.
- Không có thay đổi nào dựa trên kết quả test, kết quả chấm bởi người, hay kết quả của bộ xác
  nhận. Nếu phát hiện lỗi lập trình, sửa lỗi được ghi ở mục 9 kèm ảnh hưởng lên mọi bộ dữ liệu,
  và kết quả trước–sau đều được báo cáo.
- SBERT-kmeans và BERTopic giữ nguyên cấu hình và lưới.

## 3. Baseline chính: TaxoGen bằng code gốc

- Code: github.com/franticnerd/taxogen, commit 929a7534. Chỉ sửa những gì để chạy được (Python 3
  bằng 2to3, đường dẫn, `posix_memalign` → `malloc` trong `word2vec.c`), như
  `outputs/dblp/fidelity/original_code_changes.diff`; môi trường Python 3.7.9, numpy 1.16.6,
  scipy 1.2.3, scikit-learn 0.20.4, spherecluster 0.1.7. Mọi thay đổi thêm phải ghi ở mục 9.
- Đầu vào cho **mọi** bộ (kể cả DBLP): `papers.txt` và `keywords.txt` sinh từ corpus của repo
  (cùng tài liệu đã tách từ, cùng danh sách thuật ngữ mà Improved dùng), **chỉ gồm tài liệu của
  tập dựng**; `index.txt`, `keyword_cnt.txt` sinh bằng tiền xử lý của chính code gốc;
  `embeddings.txt` huấn luyện bằng `word2vec.c` gốc, tham số mặc định của nó, trên cùng các tài
  liệu đó. Không dùng `embeddings.txt` của bản phát hành DBLP (đã huấn luyện trên toàn bộ dữ
  liệu, gồm cả tài liệu test).
- Tham số: chọn trên val bằng lưới filter_thre {0.15, 0.2, 0.25, 0.3} × n_cluster {4, 5, 6}
  và cùng `select_params`, dựng trên train; các tham số khác giữ giá trị DBLP của code gốc
  (n_expand 100, n_cluster_iter 2, MAX_LEVEL 3).
- Code gốc không cố định seed: cây cuối cùng chạy 5 lần trên tập dựng; báo trung bình ± độ lệch
  chuẩn. Cây dùng cho phiếu chấm là **lần chạy thứ nhất**.
- Thuật ngữ mỗi nút: đầu ra `compress.py` của code gốc (N = 10, bằng top_k của các chỉ số).
  Term coverage và doc coverage của TaxoGen gốc vì thế không so được với các phương pháp gán
  mọi thuật ngữ; bài chỉ so NPMI, subsumption, sibling overlap và các chỉ số với cây chuẩn.
- Bản viết lại (`taxorec.taxonomy.builders.TaxoGen`) không xuất hiện trong so sánh chính; code
  được giữ chỉ để tái lập vòng chấm 1 của bộ phát triển, và kiểm tra độ trung thành
  (`docs/KIEM_TRA_TAXOGEN_GOC.md`) được báo cáo cùng vòng chấm đó.

## 4. Chia dữ liệu và đọc test

- Train 20% / val 30% / test 50% (tách theo tài liệu, seed như hiện tại) cho cả bộ phát triển
  và bộ xác nhận; tập dựng = train + val.
- Bộ phát triển giữ nguyên cách chia cũ. Bộ xác nhận: test được đọc một lần cho chỉ số tự động
  và một lần cho ablation, ghi ở `eval/test_access.json`; không có `--force`.
- So sánh chính ở cùng kích thước (cắt về cây nhỏ nhất trong Improved, TaxoGen gốc,
  SBERT-kmeans), kèm cỡ đầy đủ; 5 seed (TaxoGen gốc: 5 lần chạy).

## 5. Đánh giá bởi người — vòng 2, trên bộ xác nhận

- So sánh: Improved với TaxoGen gốc (lần chạy thứ nhất), lấy mẫu từ cây đã cắt về cùng kích
  thước; 60 câu quan hệ, 50 câu từ lạc đề mỗi phương pháp; cùng cách làm phiếu, câu kiểm tra
  chú ý và quy tắc loại người chấm (≥ 2 câu kiểm tra sai) như `DE_CUONG_DANH_GIA.md`.
- Nhóm so sánh chính: (số bộ xác nhận) × 2 nhiệm vụ, hiệu chỉnh Holm ở 0.05. Kết luận cho mỗi
  so sánh là một trong ba: "Improved tốt hơn" (chênh lệch dương, p_Holm < 0.05), "TaxoGen gốc
  tốt hơn" (chênh lệch âm, p_Holm < 0.05), "chưa xác lập".
- Cờ trùng vector trả lời: dẫn tới chẩn đoán (như `outputs/dblp/human_eval/INTEGRITY_NOTE.md`)
  và báo cáo trong bài; không loại người chấm vì cờ này, và phân tích được chạy tiếp
  (`--force` được ghi lại). Các cờ khác dừng phân tích cho tới khi có giải thích.
- Người chấm: 20 người, không phải tác giả. Ưu tiên người không tham gia vòng 1; nếu có người
  tham gia cả hai vòng, số người đó được báo cáo. Mỗi người làm trên máy riêng, cửa sổ ẩn danh,
  không đăng nhập; nếu được phép mở nhiều biểu mẫu cùng lúc thì ghi rõ trước khi gửi link.
- Phiếu được đóng băng (`freeze`) trước khi gửi link; dữ liệu giữ ẩn tới khi qua kiểm tra
  toàn vẹn.

## 6. Bộ phát triển — những gì báo cáo

- Chỉ số tự động (test, cùng kích thước) cho Improved, TaxoGen gốc (chạy lại theo mục 3, trên
  tập dựng), SBERT-kmeans, BERTopic.
- Vòng chấm 1 (Fashion, DBLP; Improved với bản viết lại TaxoGen): báo cáo nguyên kết quả đã
  phân tích, gồm Holm trên 4 so sánh, các cờ toàn vẹn và lời giải thích của người tổ chức, cùng
  kết luận kiểm tra độ trung thành (bản viết lại yếu hơn code gốc về NPMI).
- Handmade: nêu rõ test đã đọc 5 lần (lần đầu ở commit 1a07c58, sau đó 3 lần với `--force`
  sau khi sửa phương pháp) và việc sửa phương pháp một phần xuất phát từ ablation trên Handmade;
  báo cáo cả kết quả lần đọc đầu và kết quả sau cùng.
- Lịch sử phát triển: phương pháp sửa sau mốc 1a07c58 (4c9c91e, 259db01), quyết định trên val
  của Fashion và DBLP, trước khi đọc test của hai bộ này và trước khi có phiếu chấm.

## 7. Gợi ý (phụ)

Giữ giao thức hiện tại (mô hình, lưới, chọn trên valid, test một lần). Chạy trên bộ phát triển
có tương tác (Fashion, Handmade) và trên bộ xác nhận nếu bộ đó có tương tác người dùng–sản
phẩm; TaxoGen gốc thay bản viết lại trong Sem+Tax(TaxoGen). Mô hình chỉ dùng taxonomy được báo
kèm chú thích về cách xếp hạng điểm hòa (đích đặt sau mọi sản phẩm cùng điểm).

## 8. Báo cáo

Mọi kết quả nêu ở trên được báo cáo, kể cả khi bất lợi cho Improved. Không bỏ bộ dữ liệu, so
sánh hay vòng chấm nào sau khi đã thấy kết quả. Repo khám phá được lưu trữ (Zenodo, có thể
hạn chế truy cập) và bài có một đoạn mô tả giai đoạn khám phá.

## 9. Thay đổi sau khi chốt

- 2026-10-28 (trước khi chọn hay tải bộ xác nhận nào): hai bộ xác nhận là một bộ thương mại
  điện tử và một bộ khoa học máy tính, chọn theo `docs/CHON_BO_XAC_NHAN.md`. Người chấm vòng 2:
  20 người **khác** 20 người của vòng 1, vẫn thuộc nhóm nghiên cứu Khoa CNTT, Trường Đại học
  Công nghệ Kỹ thuật Hưng Yên. Repo công bố: `D:\TaxoGen_Improved_Publication`,
  github.com/TranDucLongAIDHSPKTHY/Improved_TaxoGen_DS.
