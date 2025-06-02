htmx.on("htmx:afterRequest", function(evt) {
    if (evt.detail.elt.id === "chat-form") {
        evt.detail.elt.reset();
    }
    console.log(evt.detail.elt);
    console.log("The element that dispatched the request: ", evt.detail.elt);
    console.log("The XMLHttpRequest: ", evt.detail.xhr);
    console.log("The target of the request: ", evt.detail.target);
    console.log("The configuration of the AJAX request: ", evt.detail.requestConfig);
    console.log("The event that triggered the request: ", evt.detail.requestConfig.triggeringEvent);
    console.log("True if the response has a 20x status code or is marked detail.isError = false in the htmx:beforeSwap event, else false: ", evt.detail.successful);
    console.log("true if the response does not have a 20x status code or is marked detail.isError = true in the htmx:beforeSwap event, else false: ", evt.detail.failed);

    const chatLog = document.getElementById('chat-log');
    chatLog.scrollTop = chatLog.scrollHeight;

    const userMessage = document.getElementById('user-message');
    const submitMessageButton = document.getElementById('submit-message');
    userMessage.value = "";
    userMessage.disabled = false;
    submitMessageButton.disabled = false;
    userMessage.focus();
});

htmx.on("htmx:onLoadError", function(evt) {
    // htmx.find("#error-div").innerHTML = "A network error occured...";
    console.log(evt)
    console.log("ERROR!")

    const userMessage = document.getElementById('user-message');
    const submitMessageButton = document.getElementById('submit-message');
    userMessage.disabled = false;
    submitMessageButton.disabled = false;
    userMessage.focus();
  }
)

htmx.on("htmx:beforeRequest", function(evt) {


});

document.addEventListener('DOMContentLoaded', function() {

  const userMessage = document.getElementById('user-message');
  userMessage.focus()

  const chatForm = document.getElementById('chat-form');
  const chatLog = document.getElementById('chat-log'); // Reference to the chat log for scrolling
  const internalSubmitButton = document.getElementById('chat-form-submit');
  const submitMessageButton = document.getElementById('submit-message');

  function appendMessageToLog(text, sender, timestamp) {
      if (!chatLog) return;

      const messageDiv = document.createElement('div');
      messageDiv.classList.add('chat-message', sender);

      const bubbleDiv = document.createElement('div');
      bubbleDiv.classList.add('message-bubble');

      const textP = document.createElement('p');
      textP.classList.add('message-text');
      textP.textContent = text;

      // const timeSpan = document.createElement('span');
      // timeSpan.classList.add('message-timestamp');
      // timeSpan.textContent = timestamp || getCurrentTime();

      bubbleDiv.appendChild(textP);
      // bubbleDiv.appendChild(timeSpan);
      messageDiv.appendChild(bubbleDiv);
      chatLog.appendChild(messageDiv);
      chatLog.scrollTop = chatLog.scrollHeight; // Scroll to bottom
  }

  chatForm.addEventListener('submit', (event) => {
    const messageText = userMessage.value.trim();
    if (messageText !== "") {
      appendMessageToLog(messageText, 'user');
      userMessage.disabled = true;
      submitMessageButton.disabled = true;
    }
  });

  submitMessageButton.addEventListener('click', (event) => {
      event.preventDefault();
      const messageText = userMessage.value.trim();
      if (messageText !== "") {
        internalSubmitButton.click();
      }
  });

  // Initial scroll to bottom if there's pre-loaded content
  if (chatLog){
      chatLog.scrollTop = chatLog.scrollHeight;
  }

  // // Font loading check
  // if (document.fonts) {
  //     document.fonts.load('1em Inter').catch(() => {
  //         console.warn('Inter font could not be loaded. Using system fallback.');
  //     });
  // }

});
