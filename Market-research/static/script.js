// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const healthIndicator = document.getElementById('health-indicator');
const researchForm = document.getElementById('researchForm');
const useDefaultCheckbox = document.getElementById('useDefault');
const customCompetitorsDiv = document.getElementById('customCompetitors');
const defaultCompetitorsDisplay = document.getElementById('defaultCompetitorsDisplay');
const defaultCompetitorsList = document.getElementById('defaultCompetitorsList');
const addCompetitorBtn = document.getElementById('addCompetitor');
const competitorsList = document.getElementById('competitorsList');
const urlsSection = document.getElementById('urls-section');
const urlsList = document.getElementById('urlsList');
const taskStatusSection = document.getElementById('task-status');
const taskIdSpan = document.getElementById('taskId');
const taskStatusSpan = document.getElementById('taskStatus');
const taskMessageSpan = document.getElementById('taskMessage');
const progressFill = document.querySelector('.progress-fill');
const resultsSection = document.getElementById('results');
const digestText = document.getElementById('digestText');
const downloadBtn = document.getElementById('downloadBtn');
const scrapedContentSection = document.getElementById('scraped-content');
const logsSection = document.getElementById('logs');
const logText = document.getElementById('logText');
const refreshLogsBtn = document.getElementById('refreshLogs');
const notification = document.getElementById('notification');
const notificationMessage = document.querySelector('.notification-message');
const notificationClose = document.querySelector('.notification-close');

// State
let currentTaskId = null;
let statusCheckInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadDefaultCompetitors();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    researchForm.addEventListener('submit', handleFormSubmit);
    useDefaultCheckbox.addEventListener('change', toggleCustomCompetitors);
    addCompetitorBtn.addEventListener('click', addCompetitor);
    downloadBtn.addEventListener('click', downloadDigest);
    refreshLogsBtn.addEventListener('click', fetchLogs);
    notificationClose.addEventListener('click', hideNotification);
    
    // Add event listener for remove competitor buttons
    competitorsList.addEventListener('click', (e) => {
        if (e.target.classList.contains('remove-competitor') || e.target.parentElement.classList.contains('remove-competitor')) {
            e.target.closest('.competitor-item').remove();
        }
    });
}

// Load Default Competitors
async function loadDefaultCompetitors() {
    try {
        const response = await fetch(`${API_BASE_URL}/default-competitors`);
        if (response.ok) {
            const competitors = await response.json();
            displayDefaultCompetitors(competitors);
        }
    } catch (error) {
        console.error('Error loading default competitors:', error);
    }
}

// Display Default Competitors
function displayDefaultCompetitors(competitors) {
    defaultCompetitorsList.innerHTML = '';
    
    for (const [name, config] of Object.entries(competitors)) {
        const competitorItem = document.createElement('div');
        competitorItem.className = 'competitor-display-item';
        competitorItem.innerHTML = `
            <div class="competitor-info">
                <h4>${name}</h4>
                <p class="competitor-url"><i class="fas fa-link"></i> ${config.url}</p>
                <div class="competitor-details">
                    <span class="detail-item">
                        <i class="fas fa-search"></i> 
                        ${config.content_selectors.length} content selectors
                    </span>
                    <span class="detail-item">
                        <i class="fas fa-ban"></i> 
                        ${config.exclude_selectors.length} exclude selectors
                    </span>
                </div>
            </div>
        `;
        defaultCompetitorsList.appendChild(competitorItem);
    }
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            healthIndicator.innerHTML = '<span class="indicator status-healthy"><i class="fas fa-check-circle"></i> Backend is healthy</span>';
        } else {
            healthIndicator.innerHTML = '<span class="indicator status-unhealthy"><i class="fas fa-times-circle"></i> Backend is not responding</span>';
        }
    } catch (error) {
        healthIndicator.innerHTML = '<span class="indicator status-unhealthy"><i class="fas fa-times-circle"></i> Cannot connect to backend</span>';
    }
}

// Form Submit Handler
async function handleFormSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(researchForm);
    const payload = {
        model_name: formData.get('model_name')
    };
    
    if (!useDefaultCheckbox.checked) {
        payload.competitors = getCompetitorsData();
        if (Object.keys(payload.competitors).length === 0) {
            showNotification('Please add at least one competitor', 'error');
            return;
        }
    }
    
    try {
        showNotification('Starting research task...', 'info');
        const response = await fetch(`${API_BASE_URL}/start-research`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (response.ok) {
            const data = await response.json();
            currentTaskId = data.task_id;
            showTaskStatus(data);
            startStatusChecking();
            showNotification('Research task started successfully!', 'success');
        } else {
            const error = await response.json();
            showNotification(`Failed to start research: ${error.detail}`, 'error');
        }
    } catch (error) {
        showNotification('Error starting research task', 'error');
    }
}

// Toggle Custom Competitors
function toggleCustomCompetitors() {
    if (useDefaultCheckbox.checked) {
        customCompetitorsDiv.classList.add('hidden');
        defaultCompetitorsDisplay.classList.remove('hidden');
    } else {
        customCompetitorsDiv.classList.remove('hidden');
        defaultCompetitorsDisplay.classList.add('hidden');
    }
}

// Add Competitor
function addCompetitor() {
    const competitorItem = document.createElement('div');
    competitorItem.className = 'competitor-item';
    competitorItem.innerHTML = `
        <input type="text" placeholder="Name" class="competitor-name">
        <input type="url" placeholder="URL" class="competitor-url">
        <textarea placeholder="Content selectors (comma separated)" class="competitor-content">main, .content, article</textarea>
        <textarea placeholder="Exclude selectors (comma separated)" class="competitor-exclude">nav, footer, .sidebar</textarea>
        <button type="button" class="remove-competitor"><i class="fas fa-trash"></i></button>
    `;
    competitorsList.appendChild(competitorItem);
}

// Get Competitors Data
function getCompetitorsData() {
    const competitors = {};
    const competitorItems = competitorsList.querySelectorAll('.competitor-item');
    
    competitorItems.forEach(item => {
        const name = item.querySelector('.competitor-name').value.trim();
        const url = item.querySelector('.competitor-url').value.trim();
        const contentSelectors = item.querySelector('.competitor-content').value.split(',').map(s => s.trim());
        const excludeSelectors = item.querySelector('.competitor-exclude').value.split(',').map(s => s.trim());
        
        if (name && url) {
            competitors[name] = {
                url: url,
                content_selectors: contentSelectors,
                exclude_selectors: excludeSelectors
            };
        }
    });
    
    return competitors;
}

// Show URLs Being Scraped
function showScrapedUrls(urls) {
    if (!urls || Object.keys(urls).length === 0) {
        urlsSection.classList.add('hidden');
        return;
    }
    
    urlsSection.classList.remove('hidden');
    urlsList.innerHTML = '';
    
    for (const [name, url] of Object.entries(urls)) {
        const urlItem = document.createElement('div');
        urlItem.className = 'url-item';
        urlItem.innerHTML = `
            <div class="url-info">
                <h4>${name}</h4>
                <a href="${url}" target="_blank" class="url-link">
                    <i class="fas fa-external-link-alt"></i> ${url}
                </a>
            </div>
            <div class="url-status">
                <i class="fas fa-spinner fa-spin"></i> Scraping...
            </div>
        `;
        urlsList.appendChild(urlItem);
    }
}

// Show Task Status
function showTaskStatus(data) {
    taskIdSpan.textContent = data.task_id;
    taskStatusSpan.textContent = data.status;
    taskMessageSpan.textContent = data.message;
    taskStatusSection.classList.remove('hidden');
    updateProgress(data.status);
    
    if (data.status === 'completed' || data.status === 'failed') {
        logsSection.classList.remove('hidden');
        fetchLogs();
        
        // Update URL status to completed
        const urlItems = urlsList.querySelectorAll('.url-item .url-status');
        urlItems.forEach(status => {
            if (data.status === 'completed') {
                status.innerHTML = '<i class="fas fa-check-circle" style="color: green;"></i> Completed';
            } else {
                status.innerHTML = '<i class="fas fa-times-circle" style="color: red;"></i> Failed';
            }
        });
    }
}

// Update Progress Bar
function updateProgress(status) {
    const progressMap = {
        'pending': 20,
        'running': 60,
        'completed': 100,
        'failed': 100
    };
    
    progressFill.style.width = `${progressMap[status] || 0}%`;
}

// Start Status Checking
function startStatusChecking() {
    statusCheckInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/research-status/${currentTaskId}`);
            if (response.ok) {
                const data = await response.json();
                showTaskStatus(data);
                
                if (data.status === 'completed') {
                    clearInterval(statusCheckInterval);
                    fetchResults();
                } else if (data.status === 'failed') {
                    clearInterval(statusCheckInterval);
                    showNotification('Research task failed', 'error');
                }
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }, 3000);
}

// Fetch Results
async function fetchResults() {
    try {
        const response = await fetch(`${API_BASE_URL}/research-result/${currentTaskId}`);
        if (response.ok) {
            const data = await response.json();
            displayResults(data);
            
            // Show scraped URLs if available
            if (data.scraped_urls) {
                showScrapedUrls(data.scraped_urls);
            }
        } else {
            showNotification('Failed to fetch results', 'error');
        }
    } catch (error) {
        showNotification('Error fetching results', 'error');
    }
}

// Display Results
function displayResults(data) {
    resultsSection.classList.remove('hidden');
    digestText.textContent = data.digest || 'No digest content available';
    
    // Show scraped content if available
    if (data.scraped_content && Object.keys(data.scraped_content).length > 0) {
        scrapedContentSection.classList.remove('hidden');
        
        const tabButtons = scrapedContentSection.querySelector('.tab-buttons');
        const tabContent = scrapedContentSection.querySelector('.tab-content');
        
        // Clear existing tabs
        tabButtons.innerHTML = '';
        tabContent.innerHTML = '';
        
        // Create tabs for each competitor
        let firstTab = true;
        for (const [company, content] of Object.entries(data.scraped_content)) {
            // Create tab button
            const tabButton = document.createElement('button');
            tabButton.className = `tab-button ${firstTab ? 'active' : ''}`;
            tabButton.textContent = company;
            tabButton.dataset.tab = company.replace(/\s+/g, '-');
            tabButton.addEventListener('click', () => switchTab(company.replace(/\s+/g, '-')));
            tabButtons.appendChild(tabButton);
            
            // Create tab content
            const tabPane = document.createElement('div');
            tabPane.className = `tab-pane ${firstTab ? 'active' : ''}`;
            tabPane.id = `tab-${company.replace(/\s+/g, '-')}`;
            
            // Create competitor header with stats
            const competitorHeader = document.createElement('div');
            competitorHeader.className = 'competitor-header';
            competitorHeader.innerHTML = `
                <h3 class="competitor-title">${company}</h3>
                <div class="content-stats">
                    <div class="stat-item">
                        <i class="fas fa-file-alt"></i>
                        <span>${content.length} characters</span>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-calculator"></i>
                        <span>${content.split(' ').length} words</span>
                    </div>
                </div>
            `;
            
            // Create content area
            const contentArea = document.createElement('pre');
            contentArea.textContent = content;
            
            // Create download button
            const downloadButton = document.createElement('button');
            downloadButton.className = 'download-scraped';
            downloadButton.innerHTML = '<i class="fas fa-download"></i> Download Content';
            downloadButton.addEventListener('click', () => downloadScrapedContent(company, content));
            
            // Assemble the tab pane
            tabPane.appendChild(competitorHeader);
            tabPane.appendChild(contentArea);
            tabPane.appendChild(downloadButton);
            
            tabContent.appendChild(tabPane);
            
            firstTab = false;
        }
    } else {
        // Show empty state if no scraped content
        scrapedContentSection.classList.remove('hidden');
        const tabContent = scrapedContentSection.querySelector('.tab-content');
        tabContent.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <p>No scraped content available</p>
            </div>
        `;
    }
    
    if (data.error) {
        showNotification(`Error: ${data.error}`, 'error');
    }
}

// Switch Tab
function switchTab(tabId) {
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        if (button.dataset.tab === tabId) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
    
    // Update tab panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        if (pane.id === `tab-${tabId}`) {
            pane.classList.add('active');
        } else {
            pane.classList.remove('active');
        }
    });
}

// Download Digest
function downloadDigest() {
    const content = digestText.textContent;
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `market_research_digest_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

// Download Scraped Content
function downloadScrapedContent(company, content) {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${company.replace(/\s+/g, '_')}_scraped_content_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

// Fetch Logs with enhanced display
async function fetchLogs() {
    try {
        const response = await fetch(`${API_BASE_URL}/logs`);
        const data = await response.json();
        
        if (data.logs) {
            displayLogs(data.logs);
        } else if (data.error) {
            logText.innerHTML = `<div class="log-error">Error: ${data.error}</div>`;
        }
    } catch (error) {
        logText.innerHTML = `<div class="log-error">Failed to fetch logs: ${error.message}</div>`;
    }
}

function displayLogs(logs) {
    if (!logs || logs.length === 0) {
        logText.innerHTML = '<div class="log-info">No logs available yet. Start a research task to see detailed logs.</div>';
        return;
    }
    
    const logHtml = logs.map(log => {
        const levelClass = `log-${log.level.toLowerCase()}`;
        const icon = getLogIcon(log.level);
        
        return `
            <div class="log-entry ${levelClass}">
                <span class="log-timestamp">${log.timestamp}</span>
                <span class="log-level">
                    <i class="${icon}"></i>
                    ${log.level}
                </span>
                <span class="log-message">${escapeHtml(log.message)}</span>
            </div>
        `;
    }).join('');
    
    logText.innerHTML = logHtml;
    
    // Auto-scroll to bottom
    logText.scrollTop = logText.scrollHeight;
}

function getLogIcon(level) {
    switch (level.toLowerCase()) {
        case 'info': return 'fas fa-info-circle';
        case 'warning': return 'fas fa-exclamation-triangle';
        case 'error': return 'fas fa-times-circle';
        case 'debug': return 'fas fa-bug';
        default: return 'fas fa-circle';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add clear logs functionality
function addClearLogsButton() {
    const logsSection = document.getElementById('logs');
    const refreshBtn = document.getElementById('refreshLogs');
    
    // Create clear logs button
    const clearBtn = document.createElement('button');
    clearBtn.id = 'clearLogs';
    clearBtn.className = 'clear-btn';
    clearBtn.innerHTML = '<i class="fas fa-trash"></i> Clear Logs';
    clearBtn.addEventListener('click', clearLogs);
    
    // Insert after refresh button
    refreshBtn.parentNode.insertBefore(clearBtn, refreshBtn.nextSibling);
}

async function clearLogs() {
    try {
        const response = await fetch(`${API_BASE_URL}/logs`, {
            method: 'DELETE'
        });
        const data = await response.json();
        
        if (data.message) {
            logText.innerHTML = '<div class="log-info">Logs cleared successfully.</div>';
        } else if (data.error) {
            logText.innerHTML = `<div class="log-error">Error clearing logs: ${data.error}</div>`;
        }
    } catch (error) {
        logText.innerHTML = `<div class="log-error">Failed to clear logs: ${error.message}</div>`;
    }
}

// Auto-refresh logs during research
let logRefreshInterval;

function startLogAutoRefresh() {
    // Refresh logs every 2 seconds during research
    logRefreshInterval = setInterval(fetchLogs, 2000);
}

function stopLogAutoRefresh() {
    if (logRefreshInterval) {
        clearInterval(logRefreshInterval);
        logRefreshInterval = null;
    }
}

// Update the research monitoring function
async function checkResearchStatus(taskId) {
    try {
        const response = await fetch(`${API_BASE_URL}/research-status/${taskId}`);
        const status = await response.json();
        
        updateProgress(status.message);
        
        if (status.status === 'completed') {
            stopLogAutoRefresh();
            // ... existing completion code ...
        } else if (status.status === 'failed') {
            stopLogAutoRefresh();
            // ... existing error code ...
        } else {
            // Continue checking and refresh logs
            setTimeout(() => checkResearchStatus(taskId), 2000);
        }
    } catch (error) {
        stopLogAutoRefresh();
        // ... existing error handling ...
    }
}

// Update start research function
async function startResearch() {
    // ... existing code ...
    
    // Start auto-refreshing logs
    startLogAutoRefresh();
    
    // ... rest of existing code ...
}

// Historical Data Management
let historicalViewMode = 'sections'; // 'sections' or 'raw'

async function fetchHistoricalData() {
    try {
        const response = await fetch('/historical-data');
        const data = await response.json();
        
        if (data.success) {
            displayHistoricalData(data);
            updateHistoricalStats(data);
        } else {
            document.getElementById('historicalSections').innerHTML = 
                `<div class="error-message">Error: ${data.error}</div>`;
        }
    } catch (error) {
        console.error('Error fetching historical data:', error);
        document.getElementById('historicalSections').innerHTML = 
            `<div class="error-message">Error fetching historical data: ${error.message}</div>`;
    }
}

function displayHistoricalData(data) {
    const sectionsContainer = document.getElementById('historicalSections');
    const rawContainer = document.getElementById('historicalRawText');
    
    if (data.sections && data.sections.length > 0) {
        // Display sections view
        sectionsContainer.innerHTML = data.sections.map(section => `
            <div class="historical-section" data-section-id="${section.id}">
                <div class="section-header" onclick="toggleSection(${section.id})">
                    <h3>${section.title}</h3>
                    <button class="expand-btn">
                        <i class="fas fa-chevron-down"></i>
                    </button>
                </div>
                <div class="section-content collapsed" id="section-${section.id}">
                    <div class="section-preview">${section.content}</div>
                    <div class="section-full hidden">${section.full_content}</div>
                    <button class="show-more-btn" onclick="showFullContent(${section.id})">
                        Show Full Content
                    </button>
                </div>
            </div>
        `).join('');
        
        // Display raw data
        rawContainer.textContent = data.sections.map(s => s.full_content).join('\n\n---\n\n');
    } else {
        sectionsContainer.innerHTML = '<div class="no-data">No historical data available</div>';
        rawContainer.textContent = 'No historical data available';
    }
}

function updateHistoricalStats(data) {
    const statsElement = document.getElementById('historicalStats');
    if (data.success) {
        statsElement.innerHTML = `
            <i class="fas fa-info-circle"></i>
            Total Sections: ${data.total_sections} | 
            Data Size: ${formatBytes(data.raw_data_length)} | 
            Last Updated: ${new Date(data.last_updated).toLocaleString()}
        `;
    } else {
        statsElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error loading data';
    }
}

function toggleSection(sectionId) {
    const sectionContent = document.getElementById(`section-${sectionId}`);
    const expandBtn = sectionContent.parentElement.querySelector('.expand-btn i');
    
    if (sectionContent.classList.contains('collapsed')) {
        sectionContent.classList.remove('collapsed');
        expandBtn.classList.remove('fa-chevron-down');
        expandBtn.classList.add('fa-chevron-up');
    } else {
        sectionContent.classList.add('collapsed');
        expandBtn.classList.remove('fa-chevron-up');
        expandBtn.classList.add('fa-chevron-down');
    }
}

function showFullContent(sectionId) {
    const sectionContent = document.getElementById(`section-${sectionId}`);
    const preview = sectionContent.querySelector('.section-preview');
    const full = sectionContent.querySelector('.section-full');
    const btn = sectionContent.querySelector('.show-more-btn');
    
    if (full.classList.contains('hidden')) {
        preview.classList.add('hidden');
        full.classList.remove('hidden');
        btn.textContent = 'Show Preview';
    } else {
        preview.classList.remove('hidden');
        full.classList.add('hidden');
        btn.textContent = 'Show Full Content';
    }
}

function toggleHistoricalView() {
    const sectionsView = document.getElementById('historicalSections');
    const rawView = document.getElementById('historicalRaw');
    const toggleBtn = document.getElementById('toggleHistoricalView');
    
    if (historicalViewMode === 'sections') {
        sectionsView.classList.add('hidden');
        rawView.classList.remove('hidden');
        toggleBtn.innerHTML = '<i class="fas fa-th-list"></i> Sections View';
        historicalViewMode = 'raw';
    } else {
        sectionsView.classList.remove('hidden');
        rawView.classList.add('hidden');
        toggleBtn.innerHTML = '<i class="fas fa-list"></i> Raw View';
        historicalViewMode = 'sections';
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

document.addEventListener('DOMContentLoaded', function() {
    // ... existing code ...
    
    // Historical data event listeners
    const refreshHistoricalBtn = document.getElementById('refreshHistorical');
    const toggleHistoricalBtn = document.getElementById('toggleHistoricalView');
    
    if (refreshHistoricalBtn) {
        refreshHistoricalBtn.addEventListener('click', fetchHistoricalData);
    }
    
    if (toggleHistoricalBtn) {
        toggleHistoricalBtn.addEventListener('click', toggleHistoricalView);
    }
    
    // Load historical data on page load
    fetchHistoricalData();
});

function showNotification(message, type = 'info') {
    notificationMessage.textContent = message;
    notification.className = `notification notification-${type}`;
    notification.classList.remove('hidden');
    
    setTimeout(() => {
        hideNotification();
    }, 5000);
}

// Hide Notification
function hideNotification() {
    notification.classList.add('hidden');
}