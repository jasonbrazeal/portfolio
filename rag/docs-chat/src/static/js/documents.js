// Provided documents.js code
const MAX_FILESIZE_MB = 100;

function validateFiles(files) {
  let sizeErrors = false;
  let typeErrors = false;
  let otherErrors = false;
  let problemFiles = [];
  const existingDocumentNodes = document.querySelectorAll("#doc-table-body tr td:first-child");
  const existingDocuments = Array.from(existingDocumentNodes).map(doc => doc.textContent.trim());
  console.log('validating files:');
  for (const file of files) {
    console.log(file);
    const fileName = file.name;
    const fileSize = file.size;
    const fileType = file.type;
    const sizeInMB = Number.parseFloat(fileSize / (1024 * 1024)).toFixed(2);

    console.log(fileName);
    console.log(fileSize);
    console.log(fileType);
    console.log(sizeInMB);

    if (sizeInMB > MAX_FILESIZE_MB) {
      sizeErrors = true;
      problemFiles.push(fileName);
      console.log(`${fileName} is too large: ${sizeInMB}MB`);
    }
    if ((fileType !== "application/pdf") && (fileType !== "text/plain")) {
      typeErrors = true;
      if (!problemFiles.includes(fileName)) {
        problemFiles.push(fileName);
      }
      console.log(`fileType is unacceptable: ${fileType || "cannot determine"}`);
    }
    if (existingDocuments.includes(fileName)) {
      otherErrors = true;
      if (!problemFiles.includes(fileName)) {
         problemFiles.push(fileName);
      }
      console.log(`${fileName} has already been uploaded`);
    }
  }

  if (!sizeErrors && !typeErrors && !otherErrors) {
    return ""
  }

  const fileSizeErrorMessage = `Individual pdfs must be < ${MAX_FILESIZE_MB}MB in size.`;
  const fileTypeErrorMessage = "Only pdf and txt files are supported at this time.";
  const fileNameErrorMessage = "File name already exists.";

  let errorMessage = "";
  if (problemFiles.length > 0) {
      errorMessage = `Problem uploading: ${problemFiles.join(", ")}.`;
      if (sizeErrors) errorMessage += ` ${fileSizeErrorMessage}`;
      if (typeErrors) errorMessage += ` ${fileTypeErrorMessage}`;
      if (otherErrors) errorMessage += ` ${fileNameErrorMessage}`;
  }
  return errorMessage.trim();
}

function dropHandler(ev) {
  ev.preventDefault();
  ev.stopPropagation();
  console.log("File(s) dropped");
  const spinner = document.getElementById('backdrop');
  spinner.classList.remove('hide');

  const form = document.getElementById('document-form');
  const input = document.getElementById('documents');
  input.files = ev.dataTransfer.files;
  const error = validateFiles(input.files);
  if (error) {
    alert(error);
    input.value = '';
    spinner.classList.add("hide");
  } else {
    form.submit();
  }
  const dropZone = document.getElementById('drop-zone');
  dropZone.classList.remove('drop-zone-dragging');
}

function dragOnHandler(ev) {
  ev.preventDefault();
  ev.stopPropagation();
  const dropZone = document.getElementById('drop-zone');
  dropZone.classList.add('drop-zone-dragging');
}

function dragOffHandler(ev) {
  ev.preventDefault();
  ev.stopPropagation();
  const dropZone = document.getElementById('drop-zone');
  dropZone.classList.remove('drop-zone-dragging');
}

function clearDocsSubmit() {
  const clearDocsForm = document.getElementById('clear-docs-form');
  clearDocsForm.submit();
  console.log("Clear documents submitted");
}

function addFileToTable(fileName, uploadedDate) {
    const tableBody = document.getElementById('doc-table-body');
    if (tableBody) {
        const newRow = tableBody.insertRow();
        const cell1 = newRow.insertCell(0);
        const cell2 = newRow.insertCell(1);
        cell1.textContent = fileName;
        cell2.textContent = uploadedDate;
    }
}

document.addEventListener('DOMContentLoaded', function() {
  const clearDocsModalElem = document.getElementById('clear-docs-modal');
  const clearDocsSubmitButton = document.getElementById('clear-docs-submit');
  const clearDocsCancelButton = document.getElementById('clear-docs-cancel');
  const clearAllTrigger = document.getElementById('clear-all-trigger');

  clearDocsSubmitButton.addEventListener('click', () => {
      clearDocsSubmit();
      closeModal(clearDocsModalElem);
  });
  clearDocsCancelButton.addEventListener('click', () => {
      closeModal(clearDocsModalElem);
  });

  const dropZone = document.getElementById('drop-zone');
  if (dropZone) {
    const events = ['drag', 'dragstart', 'dragend', 'dragover', 'dragenter', 'dragleave', 'drop'];
    events.forEach(event => {
      dropZone.addEventListener(event, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });
    dropZone.addEventListener('dragover', dragOnHandler);
    dropZone.addEventListener('dragenter', dragOnHandler);
    dropZone.addEventListener('dragleave', dragOffHandler);
    dropZone.addEventListener('dragend', dragOffHandler);
    dropZone.addEventListener('drop', dropHandler);
  }

  const form = document.getElementById('document-form');
  const button = document.getElementById('file-input-button');
  const input = document.getElementById('documents');
  const spinner = document.getElementById('backdrop');

  button.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    input.click();
  });

  input.addEventListener('change', (e) => {
    e.preventDefault();
    e.stopPropagation();
    spinner.classList.remove('hide');
    const error = validateFiles(input.files);
    if (error) {
      alert(error);
      input.value = '';
      spinner.classList.add('hide');
    } else {
      console.log("Submitting files:", input.files);
      form.submit();
    }
  });
});
