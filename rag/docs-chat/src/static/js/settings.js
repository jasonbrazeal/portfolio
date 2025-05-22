htmx.on("htmx:afterRequest", function(evt) {
  const checkButton = document.getElementById('api-key-check');
  const deleteButton = document.getElementById('delete-button');
  const apiKeyForm = document.getElementById('api-key-form');
  const apiKey = document.getElementById('api-key');

  if (evt.detail.elt.id === "api-key-submit") {
      apiKeyForm.reset();
      console.log('api key!')
      console.log(apiKey.innerText);
  }

  if (apiKey.innerText == 'None') {
    deleteButton.disabled = true;
    checkButton.disabled = true;
  } else {
    deleteButton.disabled = false;
    checkButton.disabled = false;
  }

  console.log(evt.detail.elt);
  console.log("The element that dispatched the request: ", evt.detail.elt);
  console.log("The XMLHttpRequest: ", evt.detail.xhr);
  console.log("The target of the request: ", evt.detail.target);
  console.log("The configuration of the AJAX request: ", evt.detail.requestConfig);
  console.log("The event that triggered the request: ", evt.detail.requestConfig.triggeringEvent);
  console.log("True if the response has a 20x status code or is marked detail.isError = false in the htmx:beforeSwap event, else false: ", evt.detail.successful);
  console.log("true if the response does not have a 20x status code or is marked detail.isError = true in the htmx:beforeSwap event, else false: ", evt.detail.failed);
});

htmx.on("htmx:onLoadError", function(evt) {
    // htmx.find("#error-div").innerHTML = "A network error occured...";
    console.log(evt)
    console.log("ERROR!")
  }
)

htmx.on("htmx:beforeRequest", function(evt) {
  if (evt.detail.requestConfig.verb === 'post') {
    const apiKeyInput = document.getElementById('api-key-input');
    if (apiKeyInput) {
      if (!apiKeyInput.value) {
        evt.preventDefault();
      }
    }
  }
});

document.addEventListener('DOMContentLoaded', function() {

  const checkButton = document.getElementById('api-key-check');
  const cancelButton = document.getElementById('api-key-cancel');
  const deleteButton = document.getElementById('delete-button');
  const setKeyButton = document.getElementById('set-key-button');
  const apiKeyForm = document.getElementById('api-key-form');
  const apiKeyInput = document.getElementById('api-key-input');
  const apiKeySubmit = document.getElementById('api-key-submit');

  if (document.getElementById('api-key').innerText == 'None') {
    deleteButton.disabled = true;
    checkButton.disabled = true;
  }

  cancelButton.addEventListener('click', (e) => {
    apiKeyInput.value = '';
  });

  setKeyButton.addEventListener('click', (e) => {
    apiKeyInput.focus();
  });

  // if (apiKeyForm) {
  apiKeyForm.addEventListener('keypress', (e) => {
    if (e.code === 'Enter') {
      e.preventDefault();
      apiKeySubmit.click();
    }
  });
  // }

});
