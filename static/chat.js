function updateSystemMessage(systemMessage) {

  /*
  if (
    systemMessage &&
    (!systemMessageRef || systemMessage !== systemMessageRef.content)
  ) {
    let systemMessageIndex = messages.findIndex((message) => message.role === "system");
    // If the system message exists in array, remove it
    if (systemMessageIndex !== -1) {
      messages.splice(systemMessageIndex, 1);
    }
    systemMessageRef = { role: "system", content: systemMessage };
    messages.push(systemMessageRef);
  }
  */ 
}

async function postRequest() {
  const formData = new FormData();
  
  const fileInput = document.getElementById("file-upload").files[0];
  if (fileInput) {
      formData.append("file", fileInput);
  }
  
  formData.append("messages", JSON.stringify(messages));
  
  const response = await fetch("/gpt4", {
      method: "POST",
      body: formData,
  });
  
  if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail);
  }

  return response;
}


async function handleResponse(response, messageText) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let assistantMessage = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      messages.push({
        role: "assistant",
        content: assistantMessage,
      });
      userInputElem.disabled = false; // enable textarea
      submitBtn.disabled = false;    // enable button
      break;
    }

    const text = decoder.decode(value);
    assistantMessage += text;
    messageText.innerHTML = window.renderMarkdown(assistantMessage).trim();
    highlightCode(messageText);
    autoScroll();
  }
}

window.onload = function () {
  document.getElementById("chat-form").addEventListener("submit", async function (event) {
    event.preventDefault();

    userInputElem.disabled = true;   // disable textarea
    submitBtn.disabled = true;       // disable button
    
    let userInput = userInputElem.value.trim();
    
    updateSystemMessage("");

    messages.push({ role: "user", content: userInput });
    addMessageToDiv("user", userInput);
    userInputElem.value = "";

    let messageText = addMessageToDiv("assistant");

    try {
      const response = await postRequest();
      handleResponse(response, messageText);
    } catch (error) {
      // Display the error message in a new assistant message
      addMessageToDiv("assistant", error.message);
      userInputElem.disabled = false; // re-enable textarea
      submitBtn.disabled = false;    // re-enable button
    }
  });
};



