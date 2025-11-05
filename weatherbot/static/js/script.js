// =================================================================
// GLOBAL VARIABLES
// =================================================================
let map;
let currentMarker;
let currentChart, forecastChart, historicalChart, marineChart, airChart;
let currentLocation = ''; // Stores the name of the last successfully searched location
let conversationId = null;

// =================================================================
// INITIALIZATION
// =================================================================
document.addEventListener('DOMContentLoaded', function() {
    initMap();
    initCharts();

    // Add initial greeting message to the chat
    addMessageToChat("Hello! I'm your Global Weather Assistant.", 'bot', true);

    // --- Primary Event Listeners ---
    document.getElementById('searchBtn').addEventListener('click', handleUserRequest);
    document.getElementById('locationInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault(); // Prevent form submission
            handleUserRequest();
        }
    });

    // --- Event Listeners for On-Demand Tab Loading ---
    document.getElementById('getHistoricalBtn').addEventListener('click', getHistoricalWeatherData);
    document.getElementById('marine-tab').addEventListener('shown.bs.tab', getMarineWeatherData);
    document.getElementById('air-tab').addEventListener('shown.bs.tab', getAirQualityData);
});

// =================================================================
// CORE LOGIC
// =================================================================

/**
 * Handles the entire user request flow: sending to the backend,
 * receiving the response, and updating the entire UI.
 */
async function handleUserRequest() {
    const locationInput = document.getElementById('locationInput');
    const query = locationInput.value.trim();
    if (!query) return;

    // 1. Update chat with user's message
    addMessageToChat(query, 'user', false);
    locationInput.value = '';
    
    // 2. Reset UI for the new request
    showLoading(true);
    document.getElementById('weatherResults').style.display = 'none';
    showError(null);

    try {
        // 3. Fetch data from the backend
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: query, conversation_id: conversationId })
        });
        
        const data = await response.json();
        
        if (!response.ok || data.error) {
            throw new Error(data.error || 'The server returned an unexpected error.');
        }

        conversationId = data.conversation_id;

        // 4. Render the AI's formatted response in the chat bubble
        addMessageToChat(data.response, 'bot', true);

        // --- START: NEWLY ADDED CODE ---
        // 5. Render the follow-up questions as clickable suggestions
        if (data.follow_up) {
            addFollowUpQuestionsToChat(data.follow_up);
        }
        // --- END: NEWLY ADDED CODE ---
        
        // 6. Populate the detailed UI with the structured visual_data
        if (data.visual_data && data.visual_data.status === 'success') {
            currentLocation = data.visual_data.location.name; // Save location for tab clicks
            updateUI(data.visual_data);
            document.getElementById('weatherResults').style.display = 'block';
        } else {
            console.warn("Visual data was not successful, but a text response was provided.");
        }
        
    } catch (error) {
        console.error('Request Failed:', error);
        const errorMessage = error.message;
        showError(errorMessage);
        addMessageToChat(`I'm sorry, I ran into a problem: ${errorMessage}`, 'bot', true);
    } finally {
        showLoading(false);
    }
}

/**
 * Adds a message to the chat display, rendering Markdown if specified.
 * @param {string} message The text content of the message.
 * @param {string} type 'user' or 'bot'.
 * @param {boolean} isMarkdown True if the message should be rendered from Markdown.
 */
function addMessageToChat(message, type, isMarkdown) {
    const chatDisplay = document.getElementById('chat-display');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;

    if (isMarkdown && window.marked) {
        messageDiv.innerHTML = marked.parse(message);
    } else {
        messageDiv.textContent = message;
    }
    
    chatDisplay.appendChild(messageDiv);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

// --- START: NEWLY ADDED FUNCTION ---
/**
 * Renders follow-up questions as clickable buttons in the chat.
 * @param {string} followUpText A string of questions separated by pipe characters '|'.
 */
function addFollowUpQuestionsToChat(followUpText) {
    const chatDisplay = document.getElementById('chat-display');
    const questions = followUpText.split('|').filter(q => q.trim().length > 0);

    questions.forEach(question => {
        // Create a wrapper div that looks like a message bubble
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message follow-up-message'; // Special class for styling

        // Create the button for the question
        const button = document.createElement('button');
        button.className = 'suggestion-btn';
        button.textContent = question;

        // Add a click event listener to the button
        button.addEventListener('click', () => {
            // When clicked, send the question as a new user message
            handleSuggestionClick(question);
        });

        // Append the button to the message div and the message div to the chat
        messageDiv.appendChild(button);
        chatDisplay.appendChild(messageDiv);
    });

    chatDisplay.scrollTop = chatDisplay.scrollHeight; // Scroll to show the new buttons
}

/**
 * Handles the click event for a suggestion button.
 * @param {string} query The text of the suggestion that was clicked.
 */
function handleSuggestionClick(query) {
    const locationInput = document.getElementById('locationInput');
    locationInput.value = query; // Put the suggestion text in the input bar
    handleUserRequest(); // Immediately trigger the request
}
// --- END: NEWLY ADDED FUNCTION ---


// =================================================================
// MASTER UI UPDATE & HELPER FUNCTIONS
// =================================================================

/**
 * Orchestrates updating all primary UI components from the main data object.
 * @param {object} data The visual_data object from the API response.
 */
function updateUI(data) {
    updateMap(data.location);
    updateCurrentWeather(data.current, data.location);
    updateForecastTab(data.daily);
    updateCurrentChart(data.current);
    updateForecastChart(data.daily);
}

function showLoading(show) {
    document.getElementById('loadingSpinner').style.display = show ? 'block' : 'none';
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    if (message) {
        errorText.textContent = message;
        errorDiv.style.display = 'block';
    } else {
        errorDiv.style.display = 'none';
    }
}

function getWeatherIcon(condition) {
    if (!condition) return 'question-circle-fill';
    const c = condition.toLowerCase();
    if (c.includes('clear') || c.includes('sun')) return 'sun-fill';
    if (c.includes('partly cloudy')) return 'cloud-sun-fill';
    if (c.includes('cloud') || c.includes('overcast')) return 'cloud-fill';
    if (c.includes('rain') || c.includes('drizzle') || c.includes('shower')) return 'cloud-rain-heavy-fill';
    if (c.includes('thunder')) return 'cloud-lightning-rain-fill';
    if (c.includes('snow')) return 'cloud-snow-fill';
    if (c.includes('fog') || c.includes('mist')) return 'cloud-fog2-fill';
    return 'thermometer-half';
}

// =================================================================
// UI COMPONENT & CHART UPDATE FUNCTIONS
// =================================================================

function updateMap(location) {
    if (!map || !location) return;
    const { lat, lon, name } = location;
    map.setView([lat, lon], 10);
    if (currentMarker) map.removeLayer(currentMarker);
    currentMarker = L.marker([lat, lon]).addTo(map).bindPopup(`<b>${name}</b>`).openPopup();
}

function updateCurrentWeather(current, location) {
    if (!current || !location) return;
    document.getElementById('currentTemp').textContent = `${current.temp_c?.toFixed(1) ?? '--'}°C`;
    document.getElementById('currentLocation').textContent = location.name || '--';
    document.getElementById('currentCondition').textContent = current.condition || '--';
    document.getElementById('currentHumidity').textContent = `Humidity: ${current.humidity ?? '--'}%`;
    document.getElementById('currentWind').textContent = `Wind: ${current.wind_kmh ?? '--'} km/h`;
    document.getElementById('currentPressure').textContent = `Pressure: ${current.pressure_mb ?? '--'} hPa`;
    document.getElementById('currentVisibility').textContent = `Visibility: ${current.visibility_km?.toFixed(1) ?? '--'} km`;
}

function updateForecastTab(daily) {
    const container = document.getElementById('forecastContainer');
    if (!container || !daily) return;
    container.innerHTML = '';
    // Use the full length of the daily array, but cap at a reasonable number for the UI
    const daysToShow = Math.min(daily.length, 7); 
    daily.slice(0, daysToShow).forEach(day => {
        const dayName = new Date(day.date).toLocaleString('en-US', { weekday: 'long' });
        const icon = getWeatherIcon(day.condition);
        container.innerHTML += `
            <div class="col-lg col-md-4 col-6 mb-3">
                <div class="forecast-day">
                    <h5>${dayName}</h5>
                    <div class="forecast-icon"><i class="bi bi-${icon}"></i></div>
                    <p class="forecast-temp">${day.max_temp_c?.toFixed(1)}° / ${day.min_temp_c?.toFixed(1)}°</p>
                    <p class="forecast-rain"><i class="bi bi-cloud-rain"></i> ${day.chance_of_rain}%</p>
                </div>
            </div>
        `;
    });
}

function updateCurrentChart(current) {
    if (!currentChart || !current) return;
    const { temp_c, humidity, wind_kmh, pressure_mb, visibility_km } = current;
    const data = [
        Math.min(temp_c * 2, 100),
        humidity,
        Math.min(wind_kmh * 3, 100),
        Math.max(0, (pressure_mb - 950) / 0.8),
        Math.min(visibility_km * 10, 100)
    ];
    currentChart.data.datasets[0].data = data;
    currentChart.update();
}

function updateForecastChart(daily) {
    if (!forecastChart || !daily) return;
    const labels = daily.map(d => new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    forecastChart.data.labels = labels;
    forecastChart.data.datasets[0].data = daily.map(d => d.max_temp_c);
    forecastChart.data.datasets[1].data = daily.map(d => d.chance_of_rain);
    forecastChart.update();
}

// =================================================================
// ON-DEMAND DATA FETCHING FOR TABS
// =================================================================

async function getHistoricalWeatherData() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    if (!startDate || !endDate) { showError('Please select a valid start and end date.'); return; }
    if (!currentLocation) { showError('Please search for a location first before getting historical data.'); return; }
    
    const container = document.getElementById('historicalContainer');
    container.innerHTML = '<div class="text-center p-3"><div class="spinner-border text-primary" role="status"></div></div>';

    try {
        const response = await fetch('/api/historical-weather', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ location: currentLocation, start_date: startDate, end_date: endDate })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        let tableHTML = `<table class="table table-striped table-hover"><thead><tr><th>Date</th><th>Max Temp (°C)</th><th>Min Temp (°C)</th><th>Precip. (mm)</th></tr></thead><tbody>`;
        data.daily.forEach(day => {
            tableHTML += `<tr><td>${day.date}</td><td>${day.max_temp}</td><td>${day.min_temp}</td><td>${day.precipitation_sum}</td></tr>`;
        });
        tableHTML += '</tbody></table>';
        container.innerHTML = tableHTML;

        historicalChart.data.labels = data.daily.map(d => d.date);
        historicalChart.data.datasets[0].data = data.daily.map(d => d.max_temp);
        historicalChart.data.datasets[1].data = data.daily.map(d => d.min_temp);
        historicalChart.update();

    } catch (error) {
        container.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
    }
}

async function getMarineWeatherData() {
    if (!currentLocation) return;
    const container = document.getElementById('marineContainer');
    container.innerHTML = '<div class="text-center p-3"><div class="spinner-border text-primary" role="status"></div></div>';
    
    try {
        const response = await fetch('/api/marine', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ location: currentLocation })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        if (!data.daily || data.daily.length === 0) throw new Error("No marine data found. This is likely not a coastal location.");
        
        let tableHTML = `<table class="table table-striped table-hover"><thead><tr><th>Date</th><th>Max Wave (m)</th><th>Max Wind Wave (m)</th><th>Max Swell (m)</th></tr></thead><tbody>`;
        data.daily.forEach(day => {
            tableHTML += `<tr><td>${day.date}</td><td>${day.wave_height_max}</td><td>${day.wind_wave_height_max}</td><td>${day.swell_height_max}</td></tr>`;
        });
        tableHTML += '</tbody></table>';
        container.innerHTML = tableHTML;

        marineChart.data.labels = data.hourly.slice(0, 24).map(h => new Date(h.time).toLocaleTimeString([], {hour: '2-digit'}));
        marineChart.data.datasets[0].data = data.hourly.slice(0, 24).map(h => h.wave_height);
        marineChart.data.datasets[1].data = data.hourly.slice(0, 24).map(h => h.swell_height);
        marineChart.update();

    } catch (error) {
        container.innerHTML = `<div class="alert alert-warning">${error.message}</div>`;
    }
}

async function getAirQualityData() {
    if (!currentLocation) return;
    const container = document.getElementById('airQualityContainer');
    container.innerHTML = '<div class="text-center p-3"><div class="spinner-border text-primary" role="status"></div></div>';

    try {
        const response = await fetch('/api/air-quality-visual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ location: currentLocation })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        if (!data.hourly || data.hourly.length === 0) throw new Error("Air quality data is not available for this location.");

        const latest = data.hourly[data.hourly.length - 1];
        container.innerHTML = `
            <div class="alert alert-info">
                <strong>Latest Air Quality Reading:</strong><br>
                - <strong>PM2.5:</strong> ${latest.pm2_5 ?? 'N/A'} µg/m³<br>
                - <strong>PM10:</strong> ${latest.pm10 ?? 'N/A'} µg/m³<br>
                - <strong>Ozone (O₃):</strong> ${latest.ozone ?? 'N/A'} µg/m³<br>
                - <strong>Nitrogen Dioxide (NO₂):</strong> ${latest.nitrogen_dioxide ?? 'N/A'} µg/m³
            </div>`;

        const last24Hours = data.hourly.slice(-24);
        airChart.data.labels = last24Hours.map(d => new Date(d.time).toLocaleTimeString([], {hour: '2-digit'}));
        airChart.data.datasets[0].data = last24Hours.map(d => d.pm2_5);
        airChart.data.datasets[1].data = last24Hours.map(d => d.pm10);
        airChart.data.datasets[2].data = last24Hours.map(d => d.ozone);
        airChart.data.datasets[3].data = last24Hours.map(d => d.nitrogen_dioxide);
        airChart.update();

    } catch (error) {
        container.innerHTML = `<div class="alert alert-warning">${error.message}</div>`;
    }
}

// =================================================================
// MAP & CHART INITIALIZATIONS
// =================================================================

function initMap() {
    map = L.map('map').setView([20, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);
}

function initCharts() {
    const currentCtx = document.getElementById('currentChart').getContext('2d');
    currentChart = new Chart(currentCtx, { type: 'radar', data: { labels: ['Temp.', 'Humidity', 'Wind', 'Pressure', 'Visibility'], datasets: [{ label: 'Current Weather', data: [], backgroundColor: 'rgba(54, 162, 235, 0.2)', borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 2 }] }, options: { scales: { r: { suggestedMin: 0, suggestedMax: 100 } }, responsive: true, maintainAspectRatio: false } });
    const forecastCtx = document.getElementById('forecastChart').getContext('2d');
    forecastChart = new Chart(forecastCtx, { type: 'line', data: { labels: [], datasets: [{ label: 'Max Temp (°C)', data: [], borderColor: 'rgba(255, 99, 132, 1)', tension: 0.1 }, { label: 'Chance of Rain (%)', data: [], borderColor: 'rgba(54, 162, 235, 1)', tension: 0.1, yAxisID: 'y1' }] }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { position: 'left' }, y1: { position: 'right', grid: { drawOnChartArea: false } } } } });
    const historicalCtx = document.getElementById('historicalChart').getContext('2d');
    historicalChart = new Chart(historicalCtx, { type: 'bar', data: { labels: [], datasets: [{ label: 'Max Temp (°C)', data: [], backgroundColor: 'rgba(255, 99, 132, 0.5)' }, { label: 'Min Temp (°C)', data: [], backgroundColor: 'rgba(54, 162, 235, 0.5)' }] }, options: { responsive: true, maintainAspectRatio: false } });
    const marineCtx = document.getElementById('marineChart').getContext('2d');
    marineChart = new Chart(marineCtx, { type: 'line', data: { labels: [], datasets: [{ label: 'Wave Height (m)', data: [], borderColor: 'rgba(75, 192, 192, 1)', tension: 0.1 }, { label: 'Swell Height (m)', data: [], borderColor: 'rgba(153, 102, 255, 1)', tension: 0.1 }] }, options: { responsive: true, maintainAspectRatio: false } });
    const airCtx = document.getElementById('airChart').getContext('2d');
    airChart = new Chart(airCtx, { type: 'line', data: { labels: [], datasets: [{ label: 'PM2.5', data: [] }, { label: 'PM10', data: [] }, { label: 'Ozone', data: [] }, { label: 'NO₂', data: [] }] }, options: { responsive: true, maintainAspectRatio: false } });
}