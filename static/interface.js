const chatMessagesDiv = document.getElementById("chat-messages");
const userInputElem = document.getElementById("user-input");
const submitBtn = document.getElementById("submitBtn");


// State variables
let messages = [];
let systemMessageRef = null;
let autoScrollState = true;

window.onload = function () {
    submitBtn.disabled = true;
    userInputElem.disabled = true;
};



function handleInputKeydown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        document.getElementById("submitBtn").click();
    }
}

function autoScroll() {
    if (autoScrollState) {
        chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
    }
}



  function copyText(button) {
  var text = button.closest('.card').querySelector('.card-text').textContent;
  var el = document.createElement('textarea');
  el.value = text;
  document.body.appendChild(el);
  el.select();
  document.execCommand('copy');
  document.body.removeChild(el);

  // Optional: Show a message that the text was copied
  alert('Copied to clipboard!');
}


// Event listeners
document.getElementById("user-input").addEventListener("keydown", handleInputKeydown);

chatMessagesDiv.addEventListener("scroll", function () {
    const isAtBottom =
        chatMessagesDiv.scrollHeight - chatMessagesDiv.clientHeight <=
        chatMessagesDiv.scrollTop + 1;

    autoScrollState = isAtBottom;
});

window.renderMarkdown = function (content) {
    const md = new markdownit();
    return md.render(content);
};

function highlightCode(element) {
    const codeElements = element.querySelectorAll("pre code");
    codeElements.forEach((codeElement) => {
        hljs.highlightElement(codeElement);
    });
}
function addMessageToDiv(role, content = "") {
    // Create an outer container for the avatar and message
    let outerDiv = document.createElement("div");
    outerDiv.className = role === "user" ? "outer user-outer" : "outer assistant-outer";
    
    // Create an avatar image element
    let avatarImg = document.createElement("img");
    avatarImg.className = "avatar";
    
    // Set the avatar source based on role
    if (role === "user") {
        avatarImg.src = "static/img/user_avatar.png"; // Replace with your user avatar path
    } else {
        avatarImg.src = "static/img/rhino.png"; // Replace with your assistant avatar path
    }

    // Append the avatar to the outerDiv
    outerDiv.appendChild(avatarImg);

    // Create the message div and append to outerDiv
    let messageDiv = document.createElement("div");
    messageDiv.className =
        role === "user" ? "message user-message" : "message assistant-message";
    outerDiv.appendChild(messageDiv);

    let messageText = document.createElement("p");
    messageDiv.appendChild(messageText);

    if (content) {
        let renderedContent = window.renderMarkdown(content).trim();
        messageText.innerHTML = renderedContent;
        highlightCode(messageDiv);
    }

    chatMessagesDiv.appendChild(outerDiv);
    autoScroll();

    return messageText;
}




document.addEventListener('DOMContentLoaded', (event) => {

    document.getElementById('file-upload').addEventListener('change', function() {

        const maxSize = 2 * 1024 * 1024; // 5MB in bytes

        // Display selected file name
        let fileName = this.files[0].name;
        let fileSize = this.files[0].size;
        let extra_size = ""

        if (fileSize > maxSize) {
            extra_size = "<span class='large-file-warning'> Alert: File larger than 2MB detected. Processing might take longer.</span>";
        }
    
        document.getElementById('file-name-display').innerHTML = "File Selected: " + fileName + extra_size;
        // Show close button
        document.getElementById('clearBtn').classList.remove('hidden');
    });

    document.getElementById('clearBtn').addEventListener('click', function() {
        document.getElementById('file-upload').value = ''; // Clear file input
        document.getElementById('file-name-display').innerHTML = ''; // Clear displayed file name
        // Hide close button
        document.getElementById('clearBtn').classList.add('hidden');
    });

    
    var cards = document.querySelectorAll('.card');
  
    cards.forEach(function(card) {
      var seeMoreButton = card.querySelector('.card-button');
  
      seeMoreButton.addEventListener('click', function() {
        var desc = card.querySelector('.card-text');
        var expanded = this.getAttribute('aria-expanded') === 'true';
  
        if (expanded) {
          desc.style.maxHeight = '0em'; // Set to the initial max-height when collapsed
          this.textContent = 'See Prompt';
        } else {
          desc.style.maxHeight = desc.scrollHeight + 'px'; // Set to the scrollHeight when expanded
          this.textContent = 'Hide Prompt';
        }
  
        this.setAttribute('aria-expanded', !expanded);
      });
    });






});
