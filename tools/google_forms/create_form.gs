/**
 * Builds the annotation Google Form for one dataset from forms_<dataset>.csv,
 * which `python -m taxorec make-human-eval` writes next to the sheets.
 *
 * 1. Google Sheets: File > Import > Upload forms_<dataset>.csv ("Replace spreadsheet").
 *    Rename the sheet tab to the dataset name (it becomes the form title).
 * 2. Extensions > Apps Script, paste this file, save, run createForm().
 * 3. The log (View > Logs) prints the link to send and the edit link.
 *
 * Question titles start with the item id (R001, I001, ...); that is how
 * `python -m taxorec import-forms` maps the response export back to items,
 * so do not edit the titles.
 */
function createForm() {
  var sheet = SpreadsheetApp.getActiveSheet();
  var rows = sheet.getDataRange().getDisplayValues();
  var header = rows.shift();
  var col = {};
  header.forEach(function (h, i) { col[h] = i; });
  ['task', 'item_id', 'section_title', 'title', 'help', 'choices'].forEach(function (name) {
    if (!(name in col)) throw new Error('missing column: ' + name);
  });

  var form = FormApp.create('Taxonomy annotation / Đánh giá phân cấp chủ đề - ' + sheet.getName());
  form.setDescription(
    'Thí nghiệm của nhóm nghiên cứu (Khoa Công nghệ thông tin, Trường Đại học Công nghệ Kỹ thuật Hưng Yên). Tham gia tự nguyện, ' +
    'không thù lao. Làm độc lập, không trao đổi với người khác. Thời gian khoảng 20-30 phút. ' +
    'Mở link trong cửa sổ ẩn danh (Ctrl+Shift+N), không đăng nhập tài khoản Google, ' +
    'không dùng chung máy hay trình duyệt với người chấm khác.\n' +
    'A research-group study; participation is voluntary and unpaid. Work alone; about 20-30 minutes. ' +
    'Open the link in a private window, signed out, not on a device or browser another annotator uses.');
  form.setCollectEmail(false);
  form.setShowLinkToRespondAgain(false);
  form.setProgressBar(true);
  // every respondent sees the questions of each section in a different order
  // (no order effects, and answers cannot be copied position by position)
  form.setShuffleQuestions(true);

  // declining ends the form at once; `import-forms` drops such rows
  var consent = form.addMultipleChoiceItem();
  consent.setTitle('Đồng ý tham gia / Consent')
    .setHelpText('Tôi đã đọc phiếu thông tin. / I have read the information sheet.')
    .setChoices([
      consent.createChoice('Đồng ý tham gia / I agree to take part', FormApp.PageNavigationType.CONTINUE),
      consent.createChoice('Không đồng ý / I do not agree', FormApp.PageNavigationType.SUBMIT)])
    .setRequired(true);
  form.addPageBreakItem().setTitle('Mã người chấm / Annotator ID');
  var idItem = form.addTextItem()
    .setTitle('Mã người chấm / Annotator ID (A01-A20)')
    .setHelpText('Dùng mã được cấp, không ghi họ tên / Use the code you were given, not your name')
    .setRequired(true);
  idItem.setValidation(FormApp.createTextValidation()
    .setHelpText('Dạng A01 ... A99 / Format A01 ... A99')
    .requireTextMatchesPattern('^A[0-9]{2}$')
    .build());

  var lastTask = null;
  rows.forEach(function (r) {
    var task = r[col.task];
    if (task !== lastTask) {
      form.addPageBreakItem().setTitle(r[col.section_title]);
      lastTask = task;
    }
    var item = form.addMultipleChoiceItem()
      .setTitle(r[col.title])
      .setChoiceValues(String(r[col.choices]).split('|'))
      .setRequired(true);
    if (r[col.help]) item.setHelpText(r[col.help]);
  });

  Logger.log('Send this link: ' + form.getPublishedUrl());
  Logger.log('Edit link (keep private): ' + form.getEditUrl());
}
