# Kiểm tra độ trung thành của baseline TaxoGen (DBLP) — quy tắc chốt trước

Ngày chốt: 2026-10-28. Văn bản này và `tools/taxogen_fidelity.py` được commit **trước khi**
dựng bất kỳ cây nào dưới đây. Đánh giá bởi người không thay đổi (phiếu đã chấm trên cây TaxoGen
hiện tại); kết quả kiểm tra được báo cáo kèm như bằng chứng về độ tin cậy của baseline.

## 1. Câu hỏi

Reviewer 3 (RIVF 2026): TaxoGen tái hiện trên DBLP lệch xa số liệu công bố, nên lợi thế của
Improved có thể một phần do baseline yếu. TaxoGen trong repo là bản viết lại
(`taxorec.taxonomy.builders.TaxoGen`). Câu hỏi: cây của bản viết lại có kém hơn cây của code
gốc trên cùng dữ liệu không?

Đã biết trước khi chốt (chỉ đọc code, chưa chạy): word2vec cục bộ của code gốc
(`code/word2vec.c` trong repo TaxoGen) dùng CBOW, 50 chiều, 50 vòng, negative 15,
sample 1e-4, min-count 10, window 5; bản viết lại dùng skip-gram, 100 chiều, 5 vòng,
negative 10, sample 1e-3 (gensim), min-count 10, window 5. Code gốc chạy 2 vòng
clustering–CaseOLAP mỗi nút (`n_cluster_iter` 2), bản viết lại 5.

## 2. Ba cây được so

- **T_paper**: cây TaxoGen dùng trong bài (`outputs/dblp/taxonomies/taxogen.json`, seed 42,
  dựng trên train + val, `filter_thre` 0.3 chọn trên val).
- **T_orig**: code gốc github.com/franticnerd/taxogen, commit 929a7534, tham số DBLP như trong
  code (`load_dblp_params_method`: n_cluster 5, filter_thre 0.25, n_expand 100,
  n_cluster_iter 2; MAX_LEVEL 3), trên toàn bộ dữ liệu phát hành (`input/` gồm papers,
  keywords, embeddings, index, keyword_cnt; đúng file đã kiểm sha256). Chỉ sửa những gì để
  chạy được: đường dẫn dữ liệu; cú pháp Python 2 (`print`, `Queue`) ở `utils.py`,
  `compress.py` (2to3, cơ học); `word2vec.c`: `posix_memalign` → `malloc` (Windows không có;
  chỉ đổi căn lề bộ nhớ), biên dịch bằng `zig cc -O2`. Môi trường: Python 3.7.9, numpy 1.16.6,
  scipy 1.2.3, scikit-learn 0.20.4, spherecluster 0.1.7. Thuật ngữ mỗi nút: đầu ra
  `compress.py` của chính code gốc (N = 10); số tài liệu mỗi nút: `doc_ids.txt` của nút.
  Chạy một lần (code gốc không cố định seed).
- **T_C**: bản viết lại, cấu hình như T_paper, chỉ đổi word2vec cục bộ sang giá trị của code
  gốc (sg 0, vector_size 50, epochs 50, negative 15, sample 1e-4, min_count 10, window 5).

## 3. Đo

- **Trên val** (cả ba cây đều đã thấy các tài liệu val khi dựng, nên công bằng về mức tiếp
  xúc): NPMI, subsumption, sibling overlap, coverage (top_k 10), ở **cùng kích thước** (cắt
  về cây nhỏ nhất trong ba cây, `prune_to_size`) và ở cỡ đầy đủ; số nút theo tầng; độ trùng
  cụm cấp 1 (với mỗi cụm cấp 1 của T_orig, Jaccard lớn nhất với một cụm cấp 1 của cây kia,
  trên 10 thuật ngữ đầu; lấy trung bình). Không đọc tài liệu test ở bước này.
- **Biên độ**: độ lệch chuẩn của mỗi chỉ số qua 5 seed của TaxoGen bản viết lại
  (`taxonomies/seeds/taxogen@seed*.json`), cùng cách đo; biên độ m = max(2 × sd, 0.01).

## 4. Kết luận (áp dụng máy móc)

T_paper bị coi là **yếu hơn code gốc** nếu, ở cùng kích thước trên val, T_orig hơn T_paper
quá m về NPMI **hoặc** subsumption, **hoặc** thấp hơn quá m về sibling overlap. Ngược lại:
"không phát hiện bản viết lại yếu hơn". T_C so với T_paper theo cùng cách, để đo riêng ảnh
hưởng của word2vec (mô tả).

## 5. Hệ quả, chốt trước

- **Trong mọi trường hợp**: một lần đọc test bổ sung, ghi vào `eval/test_access.json` dưới bước
  riêng `taxogen-fidelity`: Improved, T_paper, T_orig, T_C ở cùng kích thước trên test, để bài
  báo cáo Improved so với TaxoGen chạy bằng code gốc. Không chọn hay đổi gì dựa trên kết quả
  này; Improved và mọi tham số đã đóng băng. Lưu ý khi đọc: T_orig dựng trên toàn bộ dữ liệu
  (gồm cả tài liệu test), nên được lợi hơn Improved trên test.
- **Nếu T_paper yếu hơn**: bài nêu rõ baseline trong đánh giá bởi người là bản viết lại kém
  hơn code gốc trên chỉ số tự động — một hạn chế của kết quả chấm bởi người trên DBLP; bảng
  chính về chỉ số tự động thêm dòng TaxoGen (code gốc). Không thu lại phiếu.
- **Nếu không**: bài nêu đã kiểm tra bản viết lại với code gốc; khoảng cách với số liệu công bố
  được quy cho khác biệt quy trình (so cùng kích thước, lấy mẫu theo tầng, người chấm khác).
- Mọi kết quả, kể cả bất lợi cho Improved, được báo cáo.

## 6. Ghi chú sau khi chốt (trước khi có số liệu)

- 2026-10-28: dựng T_C lần đầu lỗi vì bản viết lại ghi vector cục bộ vào mảng cùng số chiều với
  vector gốc (100); code gốc dùng thẳng vector cục bộ 50 chiều. Đã sửa để mảng theo số chiều
  cục bộ; với cấu hình của bài (100 = 100) kết quả không đổi. Không có số liệu nào được xem.
- Khác biệt khác thấy khi đọc code (không sửa, chỉ ghi nhận): code gốc chọn n_expand từ lân
  cận trong các từ còn lại của nút cha và không giới hạn số tài liệu cục bộ; bản viết lại chọn
  trong toàn bộ pool của nút và giới hạn 15.000 tài liệu.

## 7. Kết quả (2026-10-28)

Code gốc chạy xong (`main.py` rồi `compress.py`, mã thoát 0; khoảng 2 giờ). Ba cây đều có
155 nút (5 / 25 / 125). Mọi thuật ngữ trong 10 thuật ngữ đầu của các nút T_orig đều có trong
từ vựng corpus, nên NPMI tính trên đủ 10 thuật ngữ như hai cây kia. Số liệu:
`outputs/dblp/fidelity/compare_val.json`, `test.json`.

**Val, cùng kích thước 155 nút** (biên độ m: NPMI 0.033, subsumption 0.022, overlap 0.010):

| cây | NPMI | subsumption | sibling overlap ↓ |
|---|---|---|---|
| T_paper | −0.132 | 0.329 | 0.0197 |
| T_C | −0.064 | 0.327 | 0.0219 |
| T_orig | 0.011 | 0.203 | 0.0162 |

Kết luận theo mục 4: **T_paper yếu hơn code gốc** (NPMI: T_orig hơn 0.143 > m; subsumption và
overlap không đạt tiêu chí). Word2vec của code gốc (T_C) thu hẹp khoảng một nửa khoảng cách NPMI
(+0.068), không đổi subsumption và overlap. Độ trùng cụm cấp 1 với T_orig: 0.14 (T_paper và T_C
như nhau). Term/doc coverage của T_orig (0.10 / 0.05) không so được: cây chuyển đổi chỉ giữ
10 thuật ngữ mỗi nút (đầu ra `compress.py`), không phải toàn bộ thuật ngữ được gán.

**Test, cùng kích thước 103 nút** (đọc một lần, bước `taxogen-fidelity` trong
`eval/test_access.json`; cờ `dirty` chỉ do hai file zip export chưa theo dõi ở gốc repo):

| cây | NPMI | subsumption | sibling overlap ↓ |
|---|---|---|---|
| Improved | 0.134 | 0.378 | 0.0225 |
| T_orig | 0.130 | 0.173 | 0.0137 |
| T_C | 0.022 | 0.319 | 0.0191 |
| T_paper | −0.001 | 0.295 | 0.0182 |

T_orig dựng trên toàn bộ dữ liệu (gồm tài liệu test) và chạy một lần, không có độ lệch chuẩn.

Hệ quả theo mục 5: bài nêu baseline TaxoGen trong đánh giá bởi người trên DBLP là bản viết lại,
kém hơn code gốc về NPMI — một hạn chế của kết quả chấm bởi người trên DBLP; bảng chỉ số tự động
thêm dòng TaxoGen (code gốc). Không thu lại phiếu.
