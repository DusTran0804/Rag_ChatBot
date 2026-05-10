const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const chatBox = document.getElementById('chatBox');
const typingIndicator = document.getElementById('typingIndicator');
const clearBtn = document.getElementById('clearBtn');

let sessionId = 'web_session_' + Math.random().toString(36).substring(7);

// Initialize marked options if needed (for markdown rendering)
if(window.marked) {
    marked.setOptions({
        breaks: true,
        gfm: true
    });
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;

    // 1. Show user message
    appendMessage(query, 'user');
    userInput.value = '';
    
    // 2. Show typing indicator
    typingIndicator.classList.add('active');
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        // 3. Call API
        const response = await fetch('/api/v1/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, session_id: sessionId })
        });
        
        const data = await response.json();
        
        // 4. Hide typing, show bot message
        typingIndicator.classList.remove('active');
        
        if (response.ok) {
            appendMessage(data.response, 'bot', true);
        } else {
            appendMessage(`Lỗi: ${data.detail || 'Có lỗi xảy ra'}`, 'bot');
        }
        
    } catch (err) {
        typingIndicator.classList.remove('active');
        appendMessage(`Lỗi kết nối: ${err.message}`, 'bot');
    }
});

function appendMessage(text, sender, isMarkdown = false) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);
    
    let contentHtml = text;
    if (isMarkdown && window.marked) {
        contentHtml = marked.parse(text);
    } else {
        // Fallback if marked is not loaded
        contentHtml = text.replace(/\n/g, '<br>');
    }

    const avatarUrl = sender === 'bot' 
        ? 'https://api.dicebear.com/7.x/bottts/svg?seed=ragbot&backgroundColor=1e1e2f'
        : 'https://api.dicebear.com/7.x/avataaars/svg?seed=user&backgroundColor=2563eb';

    msgDiv.innerHTML = `
        <img src="${avatarUrl}" alt="${sender}" class="msg-avatar">
        <div class="msg-content">
            ${contentHtml}
        </div>
    `;
    
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

clearBtn.addEventListener('click', () => {
    // Keep only the first welcome message
    while (chatBox.children.length > 1) {
        chatBox.removeChild(chatBox.lastChild);
    }
    sessionId = 'web_session_' + Math.random().toString(36).substring(7); // reset session
});
