// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;
let selectedModelId = null;
let availableModels = null;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles, modelSelect, providerStatus;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    modelSelect = document.getElementById('modelSelect');
    providerStatus = document.getElementById('providerStatus');

    setupEventListeners();
    createNewSession();
    loadCourseStats();
    loadAvailableModels();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });


    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });

    // New chat button
    const newChatButton = document.getElementById('newChatButton');
    if (newChatButton) {
        newChatButton.addEventListener('click', createNewSession);
    }

    // Model selection change
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            selectedModelId = e.target.value;
            updateProviderStatus();
        });
    }
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const requestBody = {
            query: query,
            session_id: currentSessionId
        };

        // Include model_id if selected
        if (selectedModelId) {
            requestBody.model_id = selectedModelId;
        }

        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);

    let html = `<div class="message-content">${displayContent}</div>`;

    if (sources && sources.length > 0) {
        // Format sources as individual cards/badges
        const formattedSources = sources.map(source => {
            // Handle both old string format and new object format for backward compatibility
            if (typeof source === 'string') {
                return `<div class="source-item">${escapeHtml(source)}</div>`;
            }

            const displayText = escapeHtml(source.display_text || 'Unknown Source');
            const courseTitle = source.course_title ? escapeHtml(source.course_title) : '';
            const lessonNumber = source.lesson_number !== undefined ? `Lesson ${source.lesson_number}` : '';

            // Build the source card
            let sourceCard = '<div class="source-item">';

            if (source.lesson_link) {
                sourceCard += `<a href="${escapeHtml(source.lesson_link)}" target="_blank" rel="noopener noreferrer" class="source-link">`;
                sourceCard += `<div class="source-icon">📄</div>`;
                sourceCard += `<div class="source-details">`;
                if (courseTitle) {
                    sourceCard += `<div class="source-course">${courseTitle}</div>`;
                }
                if (lessonNumber) {
                    sourceCard += `<div class="source-lesson">${lessonNumber}</div>`;
                }
                sourceCard += `</div>`;
                sourceCard += `<div class="source-arrow">→</div>`;
                sourceCard += `</a>`;
            } else {
                sourceCard += `<div class="source-icon">📄</div>`;
                sourceCard += `<div class="source-details">`;
                if (courseTitle) {
                    sourceCard += `<div class="source-course">${courseTitle}</div>`;
                }
                if (lessonNumber) {
                    sourceCard += `<div class="source-lesson">${lessonNumber}</div>`;
                }
                sourceCard += `</div>`;
            }

            sourceCard += '</div>';
            return sourceCard;
        }).join('');

        html += `
            <details class="sources-collapsible" open>
                <summary class="sources-header">
                    <span class="sources-icon">📚</span>
                    <span class="sources-title">Sources</span>
                    <span class="sources-count">${sources.length}</span>
                </summary>
                <div class="sources-content">${formattedSources}</div>
            </details>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}

// Load available models
async function loadAvailableModels() {
    try {
        console.log('Loading available models...');
        const response = await fetch(`${API_URL}/models`);

        if (!response.ok) {
            throw new Error('Failed to load models');
        }

        const data = await response.json();
        console.log('Models data received:', data);

        availableModels = data;
        selectedModelId = data.default_model_id;

        // Populate dropdown
        populateModelDropdown(data);

        // Update provider status
        updateProviderStatus();

    } catch (error) {
        console.error('Error loading models:', error);
        if (modelSelect) {
            modelSelect.innerHTML = '<option value="">Error loading models</option>';
        }
        if (providerStatus) {
            providerStatus.textContent = 'Error loading models';
            providerStatus.className = 'provider-status error';
        }
    }
}

// Populate model dropdown with grouped options
function populateModelDropdown(modelsData) {
    if (!modelSelect) return;

    modelSelect.innerHTML = '';

    // Create optgroups for each provider
    modelsData.providers.forEach(provider => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = provider.display_name;

        provider.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.model_id;
            option.textContent = model.display_name;

            // Add context window info if available
            if (model.context_window) {
                option.textContent += ` (${Math.floor(model.context_window / 1000)}K)`;
            }

            // Mark default as selected
            if (model.model_id === selectedModelId) {
                option.selected = true;
            }

            optgroup.appendChild(option);
        });

        modelSelect.appendChild(optgroup);
    });
}

// Update provider status display
function updateProviderStatus() {
    if (!providerStatus || !availableModels) return;

    // Find which provider owns the selected model
    let providerName = 'Unknown';

    for (const provider of availableModels.providers) {
        for (const model of provider.models) {
            if (model.model_id === selectedModelId) {
                providerName = provider.display_name;
                break;
            }
        }
    }

    providerStatus.textContent = providerName;
    providerStatus.className = 'provider-status active';
}