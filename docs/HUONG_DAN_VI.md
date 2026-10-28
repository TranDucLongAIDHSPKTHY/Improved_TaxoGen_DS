# Hướng dẫn chạy thực nghiệm (bản xây lại, 10/2026)

Tài liệu cho các thành viên nhóm. Các quyết định đã chốt nằm ở `docs/BAN_CHOT.md`.

## 1. Cài đặt (một lần)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel   # bắt buộc: pip cũ không cài được -e .
pip install -r requirements-lock.txt      # môi trường giống hệt bản đã kiểm tra (gồm model spaCy, BERTopic)
pip install -e .
$env:TMP = "D:\tmp"; $env:TEMP = "D:\tmp"  # nếu pip báo lỗi pip-req-tracker
$env:PYTHONHASHSEED=0                      # đặt trước mỗi phiên làm việc
python -m pytest                           # phải thấy "75 passed"
```

Kiểm tra toàn bộ pipeline trên dữ liệu giả (khoảng 1 phút, không tải gì):

```powershell
python -m taxorec make-toy-data --config configs/toy.yaml
python -m taxorec all --config configs/toy.yaml
```

Kết quả ở `outputs/toy/REPORT.md`. **Số liệu toy không có ý nghĩa, không đưa vào bài.**

### Chạy trên máy khác

Mọi bước cho cùng một bài báo phải chạy trên **cùng một máy**: CPU khác nhau có thể
làm kết quả số học lệch ở chữ số cuối. Tên CPU, số luồng và RAM được ghi trong
`env.json` của mỗi lần chạy.

```powershell
git clone https://github.com/TranDucLongAIDHSPKTHY/Improved_TaxoGen.git
cd Improved_TaxoGen
# làm các lệnh cài đặt ở mục 1, rồi tải dữ liệu (thư mục data\ không có trong repo):
python -m taxorec download --config configs/amazon_fashion.yaml
python -m taxorec download --config configs/handmade_products.yaml
python -m taxorec download --config configs/dblp.yaml
```

Lệnh `download` của DBLP tải bản phát hành gốc của TaxoGen từ Google Drive (khoảng 380MB),
kiểm tra sha256 trùng với bản đã dùng khi phát triển, rồi chỉ giải nén `papers.txt`,
`keywords.txt`, `embeddings.txt` vào `data\dblp_raw\input\`. File tải về khác mã sẽ bị từ chối.

Lần chạy đầu cần Internet để tải model SBERT (đã ghim đúng phiên bản trong config).
Config đặt spaCy chạy 6 tiến trình, hợp với i5-12400F; nếu máy có nhiều nhân hơn,
thêm `--set corpus.tokenizer.n_process=8`.

## 2. Ba dataset

| dataset | config | dùng cho |
|---|---|---|
| Amazon Fashion | `configs/amazon_fashion.yaml` | taxonomy (tự động + chấm bởi người) + gợi ý |
| DBLP | `configs/dblp.yaml` | taxonomy (tự động + chấm bởi người), so với TaxoGen gốc |
| Handmade Products | `configs/handmade_products.yaml` | taxonomy (tự động, có cây danh mục chuẩn) + gợi ý |

## 3. Thứ tự chạy

`C=configs/<dataset>.yaml`; mỗi lệnh có dạng `python -m taxorec <bước> --config $C`.

| bước | lệnh | ghi chú |
|---|---|---|
| 1 | `download` | Amazon: file review + metadata; DBLP: bản phát hành TaxoGen, có kiểm tra sha256 |
| 2 | `prepare-rec` | lọc user ≥ 3 tương tác (5-core không còn dữ liệu với bản 2023), chia train/valid/test, embedding sản phẩm (chỉ metadata); không áp dụng cho DBLP |
| 3 | `prepare-corpus` | corpus review đã loại các tương tác valid/test; đọc streaming |
| 4 | `tune-taxonomy` | tài liệu chia cố định train 20% / val 30% / test 50%; dựng trên train, chọn tham số trên val → `tuning/chosen.json` |
| 5 | `build-taxonomy` | taxonomy chính (seed 42) dựng trên train + val, dùng tham số đã chọn |
| 6 | `build-taxonomy --all-seeds` | 5 seed để báo cáo trung bình ± độ lệch chuẩn |
| 6b | `sensitivity` | độ nhạy tham số: đổi từng tham số một quanh cấu hình đã chọn, dựng trên train, chấm trên val (không đọc test) |
| 7 | `ablation` | split (docs/terms) × clusterer × relation = 16 tổ hợp + 6 biến thể (không UMAP; không mỏ neo; không gắn term hiếm; không trừ vector trung bình; phân cụm term SBERT; phân cụm term word2vec), × 3 seed |
| 8 | `eval-taxonomy` | NPMI, subsumption, overlap, coverage, so với danh mục chuẩn; kết quả chính là **so cùng kích thước** (cắt mọi taxonomy về cỡ của cái nhỏ nhất, cột `*@matched_size`), kèm kết quả cỡ đầy đủ — chấm trên **test**; bước 7 và 8 mỗi bước chỉ được đọc test **một lần** (ghi ở `eval/test_access.json`), chỉ chạy khi phương pháp đã chốt |
| 9 | `make-human-eval` | phiếu chấm lấy mẫu từ các taxonomy đã cắt về cùng kích thước, `forms_<dataset>.csv`, file khoá (chỉ Fashion và DBLP) |
| 10 | (pilot) | chấm thử với 2–3 người ngoài nhóm 20; nếu phải sửa thì chạy lại bước 9 |
| 11 | `freeze` | **đóng băng** batch trước khi gửi phiếu cho 20 người |
| 12 | (chấm nhãn) | xem mục 4 |
| 13 | `import-forms --file <export.csv>` | lưu nguyên bản export, chuyển sang định dạng phân tích |
| 14 | `analyze-human-eval` | kiểm tra tính toàn vẹn → thống kê |
| 14b | `human-eval-holm --config configs/amazon_fashion.yaml --config configs/dblp.yaml` | hiệu chỉnh Holm một lần trên cả 4 so sánh chính (2 bộ × 2 nhiệm vụ); chỉ chạy khi cả hai bộ đã qua bước 14 |
| 15 | `run-rec` | các mô hình gợi ý, tinh chỉnh trên valid, test một lần |
| 16 | `report` | `outputs/<dataset>/REPORT.md` |
| 17 | `archive` | file zip kèm sha256 để đưa lên Zenodo |

Mỗi lệnh sinh một thư mục `outputs/<dataset>/runs/<bước>/<thời điểm>-<hash>/` chứa
config, phiên bản thư viện, commit git, sha256 dữ liệu đầu vào và log.
**Mọi con số trong bài phải trỏ được về một thư mục run như vậy.**

Trên máy 16GB RAM, nếu bước 3 hoặc 7 quá chậm có thể giảm tải mà vẫn ghi lại được:
`--set corpus.sample_fraction=0.5` (giữ một nửa review một cách tất định) hoặc
`--set taxonomy.ablation.seeds=[0,1,2]`. Mọi tham số override đều được lưu trong thư mục run.

## 4. Quy trình chấm bởi người

1. **Trước khi bắt đầu:** gửi phiếu thông tin `docs/human_eval/PHIEU_THONG_TIN_VA_DONG_Y.md` cho người chấm. Đây là thí nghiệm nội bộ của nhóm nghiên cứu (không có số phê duyệt đạo đức); câu đầu của Form là Đồng ý / Không đồng ý, chọn Không đồng ý thì Form kết thúc và `import-forms` bỏ qua dòng đó.
2. Cấp mã A01–A20. Bảng đối chiếu họ tên–mã chỉ bạn Long giữ, ngoại tuyến, **không** đưa vào repo hay `outputs/`.
3. Tạo Google Form cho mỗi dataset: nhập `outputs/<ds>/human_eval/batch/forms_<ds>.csv` vào Google Sheets, chạy `tools/google_forms/create_form.gs`. Kiểm tra trong cài đặt Form rằng tính năng thu email đã tắt.
4. Chấm thử với 2–3 người ngoài nhóm 20. Nếu phải sửa thì chạy lại `make-human-eval`, sau đó chạy `freeze`.
5. Gửi link Form. Mỗi người làm 2 buổi (DBLP, Amazon Fashion), mỗi buổi khoảng 20–30 phút, mở link trong cửa sổ ẩn danh, không đăng nhập, không dùng chung máy hay trình duyệt (Google Forms tự khôi phục bản nháp theo tài khoản và trình duyệt).
6. Tải file export (CSV) của từng Form, rồi chạy `import-forms --file <file>`. **Không chỉnh sửa file export.**
7. `analyze-human-eval`. Nếu có cờ `critical` thì dừng lại tìm nguyên nhân; không dùng `--force` cho số liệu đưa vào bài.
8. `File KEY_DO_NOT_SHARE.json` chỉ bạn Long giữ; người giữ khoá không tham gia chấm. Các tác giả cũng không tham gia chấm.

## 5. Không dùng lại từ bản cũ

Code, dữ liệu và kết quả của bản cũ (V10) không có trong repo này và không được
dùng lại; mọi con số của bài mới chỉ đến từ pipeline ở đây.
