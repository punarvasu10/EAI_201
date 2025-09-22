from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import heapq
import math

class CampusGraph:
    def __init__(self):
        self.locations = {
            'Main Gate': {'x': 0, 'y': 0, 'info': 'Entry point to campus, Security check available 24/7'},
            'Admin Block': {'x': 260, 'y': 100, 'info': 'Registrar office, Fee payment counter, Administrative services'},
            'Engineering Block': {'x': 400, 'y': 150, 'info': 'Engineering departments, Computer labs, Faculty offices'},
            'Guest House': {'x': 410, 'y': 250, 'info': 'Accommodation for visiting faculty and guests'},
            'Faculty Housing': {'x': 590, 'y': 300, 'info': 'Residential area for faculty members'},
            'Food Court': {'x': 630, 'y': 200, 'info': 'Cafeteria, Multiple food outlets, Open 7 AM - 10 PM'},
            'Hostel': {'x': 820, 'y': 350, 'info': 'Student accommodation, Warden office, Common rooms'},
            'Cricket Ground': {'x': 1290, 'y': 400, 'info': 'Sports facility, Cricket matches, Physical training'}
        }

        self.distances = {
            'Main Gate': {'Admin Block': 260},
            'Admin Block': {'Main Gate': 260, 'Engineering Block': 140, 'Guest House': 200},
            'Engineering Block': {'Admin Block': 140, 'Guest House': 100, 'Faculty Housing': 180, 'Food Court': 220},
            'Guest House': {'Admin Block': 200, 'Engineering Block': 100},
            'Faculty Housing': {'Engineering Block': 180, 'Food Court': 150, 'Hostel': 250},
            'Food Court': {'Engineering Block': 220, 'Faculty Housing': 150, 'Hostel': 200},
            'Hostel': {'Faculty Housing': 250, 'Food Court': 200, 'Cricket Ground': 500},
            'Cricket Ground': {'Hostel': 500}
        }

    def get_neighbors(self, location):
        return self.distances.get(location, {})

    def get_distance(self, from_loc, to_loc):
        return self.distances.get(from_loc, {}).get(to_loc, float('inf'))

    def get_path_coordinates(self, path):
        return [(self.locations[loc]['x'], self.locations[loc]['y']) for loc in path if loc in self.locations]

class BotBrain:
    def __init__(self):
        self.campus = CampusGraph()
        self.search_stats = {}

    def uniform_cost_search(self, start, goal):
        if start not in self.campus.locations or goal not in self.campus.locations:
            return None, 0

        queue = [(0, start, [start])]
        visited = set()
        nodes_explored = 0

        while queue:
            cost, current, path = heapq.heappop(queue)
            if current in visited:
                continue
                
            visited.add(current)
            nodes_explored += 1

            if current == goal:
                self.search_stats['UCS'] = {'nodes_explored': nodes_explored, 'distance': cost}
                return path, cost

            for neighbor, distance in self.campus.get_neighbors(current).items():
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + distance, neighbor, path + [neighbor]))

        return None, 0

    def get_building_info(self, building):
        return self.campus.locations.get(building, {}).get('info', 'No information available')

    def estimate_walking_time(self, distance):
        if distance == 0:
            return "0 seconds"
        time_seconds = distance / 1.39  # 5 km/h in m/s
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        return f"{minutes} minutes {seconds} seconds"

    def get_path_coordinates(self, path):
        return self.campus.get_path_coordinates(path)

class CampusBot:
    def __init__(self, bot_brain):
        self.bot_brain = bot_brain

    def process_query(self, query):
        query_lower = query.lower().strip()

        # Route queries
        route_queries = {
            ('hostel', 'main gate'): ('Main Gate', 'Hostel'),
            ('cricket ground', 'main gate'): ('Main Gate', 'Cricket Ground'),
            ('guest house', 'main gate'): ('Main Gate', 'Guest House')
        }

        for keywords, (start, end) in route_queries.items():
            if all(kw in query_lower for kw in keywords):
                path, distance = self.bot_brain.uniform_cost_search(start, end)
                if path:
                    return f"From {start} to {end}: Walk {distance} meters. Path: {' → '.join(path)}. Estimated time: {self.bot_brain.estimate_walking_time(distance)}."

        # Admin Block specific query
        if 'admin block' in query_lower:
            if any(phrase in query_lower for phrase in ['main gate', 'from main gate', 'how to go']):
                path, distance = self.bot_brain.uniform_cost_search('Main Gate', 'Admin Block')
                if path:
                    return f"From main gate, walk {distance} meters to reach Admin Block. Path: {' → '.join(path)}. Estimated time: {self.bot_brain.estimate_walking_time(distance)}."
            return "From main gate, walk straight for 260 meters to reach the Admin Block. You'll find the Registrar office, Fee payment counter, and other Administrative services there."

        # Location info queries
        if any(word in query_lower for word in ['where is', 'how to go', 'direction', 'navigate']):
            for building in self.bot_brain.campus.locations:
                if building.lower() in query_lower:
                    info = self.bot_brain.get_building_info(building)
                    path, distance = self.bot_brain.uniform_cost_search('Main Gate', building)
                    if path and len(path) > 1:
                        return f"{building}: {info}\n\nFrom Main Gate: Walk {distance} meters ({' → '.join(path)}). Estimated time: {self.bot_brain.estimate_walking_time(distance)}."
                    return f"{building}: {info}"

        # General building info
        for building in self.bot_brain.campus.locations:
            if building.lower().replace(' ', '').replace('block', '') in query_lower.replace(' ', ''):
                return f"{building}: {self.bot_brain.get_building_info(building)}"

        return "I can help you navigate the campus! Try asking: 'Where is Admin Block?', 'How to go from Main Gate to Food Court?', or ask about any campus building."

# Initialize components
bot = BotBrain()
campus_bot = CampusBot(bot)
app = Flask(__name__)
CORS(app)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BotBrain - Campus Navigator</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Arial', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .header { background: rgba(255, 255, 255, 0.95); padding: 1rem 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); position: relative; }
        .logo { position: absolute; right: 20px; top: 10px; height: 60px; }
        h1 { color: #2c3e50; font-size: 2rem; margin-bottom: 0.5rem; }
        .subtitle { color: #7f8c8d; font-size: 1.1rem; }
        .container { max-width: 1400px; margin: 2rem auto; padding: 0 1rem; display: grid; grid-template-columns: 1fr 400px; gap: 2rem; }
        .map-section, .controls { background: white; border-radius: 15px; padding: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        #map { height: 600px; border-radius: 10px; border: 3px solid #3498db; }
        .controls { height: fit-content; }
        .control-group { margin-bottom: 1.5rem; }
        label { display: block; margin-bottom: 0.5rem; font-weight: bold; color: #2c3e50; }
        select, button, input { width: 100%; padding: 0.8rem; border: 2px solid #bdc3c7; border-radius: 8px; font-size: 1rem; transition: all 0.3s ease; }
        select:focus, input:focus { outline: none; border-color: #3498db; box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1); }
        button { background: #3498db; color: white; border: none; cursor: pointer; font-weight: bold; margin-bottom: 0.5rem; }
        button:hover { background: #2980b9; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3); }
        .algorithm-btn { background: #27ae60; }
        .algorithm-btn:hover { background: #229954; }
        .results { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #3498db; }
        .chatbot { background: #2c3e50; color: white; border-radius: 15px; padding: 1.5rem; margin-top: 1rem; height: 300px; display: flex; flex-direction: column; }
        .chat-messages { flex: 1; overflow-y: auto; margin-bottom: 1rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px; }
        .message { margin-bottom: 1rem; padding: 0.5rem; border-radius: 8px; }
        .user-message { background: #3498db; text-align: right; }
        .bot-message { background: #34495e; }
        .chat-input { display: flex; gap: 0.5rem; }
        .chat-input input { flex: 1; margin-bottom: 0; }
        .chat-input button { width: auto; padding: 0.8rem 1.5rem; margin-bottom: 0; }
        @media (max-width: 1024px) { .container { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>BotBrain Campus Navigator</h1>
        <p class="subtitle">Chanakya University - Intelligent Campus Navigation System</p>
        <img src="https://chanakyauniversity.edu.in/wp-content/themes/chanakya-university/img/chanakya-university.svg" alt="Chanakya University Logo" class="logo">
    </div>

    <div class="container">
        <div class="map-section">
            <h2>Campus Map</h2>
            <div id="map"></div>
        </div>

        <div class="controls">
            <div class="control-group">
                <label for="start">From:</label>
                <select id="start">
                    <option value="">Select starting location</option>
                    <option value="Main Gate">Main Gate</option>
                    <option value="Admin Block">Admin Block</option>
                    <option value="Engineering Block">Engineering Block</option>
                    <option value="Guest House">Guest House</option>
                    <option value="Faculty Housing">Faculty Housing</option>
                    <option value="Food Court">Food Court</option>
                    <option value="Hostel">Hostel</option>
                    <option value="Cricket Ground">Cricket Ground</option>
                </select>
            </div>

            <div class="control-group">
                <label for="end">To:</label>
                <select id="end">
                    <option value="">Select destination</option>
                    <option value="Main Gate">Main Gate</option>
                    <option value="Admin Block">Admin Block</option>
                    <option value="Engineering Block">Engineering Block</option>
                    <option value="Guest House">Guest House</option>
                    <option value="Faculty Housing">Faculty Housing</option>
                    <option value="Food Court">Food Court</option>
                    <option value="Hostel">Hostel</option>
                    <option value="Cricket Ground">Cricket Ground</option>
                </select>
            </div>

            <button onclick="findPath()" class="algorithm-btn">Find Path (UCS) - Recommended</button>
            
            <div id="results" class="results" style="display:none;">
                <h3>Results</h3>
                <div id="path-info"></div>
            </div>

            <div class="chatbot">
                <h3>Campus Assistant</h3>
                <div id="chat-messages" class="chat-messages">
                    <div class="message bot-message">Hi! I'm your campus navigation assistant. Ask me about directions or building information! Try: "How to go from Main Gate to Hostel?"</div>
                </div>
                <div class="chat-input">
                    <input type="text" id="chat-input" placeholder="Ask me: Where is Admin Block?" onkeypress="handleChatKeyPress(event)">
                    <button onclick="sendChatMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map('map').setView([13.222022, 77.755447], 16);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '© OpenStreetMap contributors' }).addTo(map);

        const locations = {
            'Main Gate': [13.220103, 77.754096],
            'Admin Block': [13.222022, 77.755447],
            'Engineering Block': [13.223362, 77.755995],
            'Guest House': [13.223414, 77.754096],
            'Faculty Housing': [13.223614, 77.757218],
            'Food Court': [13.224782, 77.757207],
            'Hostel': [13.224465, 77.759143],
            'Cricket Ground': [13.228913, 77.757116]
        };

        Object.entries(locations).forEach(([name, coords]) => {
            L.marker(coords).addTo(map).bindPopup(`<strong>${name}</strong>`);
        });

        let pathPolyline = null;

        function findPath() {
            const start = document.getElementById('start').value;
            const end = document.getElementById('end').value;
            if (!start || !end) return alert('Please select both starting location and destination');

            fetch('/find_path', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({start, end}) })
                .then(r => r.json()).then(data => { displayResults(data); drawPath(data.path); })
                .catch(e => { console.error('Error:', e); alert('An error occurred while finding the path'); });
        }

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            const pathInfo = document.getElementById('path-info');
            
            if (data.path && data.path.length > 0) {
                pathInfo.innerHTML = `<strong>Algorithm:</strong> UCS (Uniform Cost Search)<br><strong>Path:</strong> ${data.path.join(' → ')}<br><strong>Distance:</strong> ${data.distance} meters<br><strong>Walking Time:</strong> ${data.walking_time}<br><strong>Nodes Explored:</strong> ${data.nodes_explored}`;
            } else {
                pathInfo.innerHTML = '<strong>No path found!</strong>';
            }
            resultsDiv.style.display = 'block';
        }

        function drawPath(path) {
            if (pathPolyline) map.removeLayer(pathPolyline);
            if (path && path.length > 0) {
                pathPolyline = L.polyline(path.map(loc => locations[loc]), { color: '#e74c3c', weight: 6, opacity: 0.8 }).addTo(map);
                map.fitBounds(pathPolyline.getBounds(), {padding: [20, 20]});
            }
        }

        function sendChatMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;

            addChatMessage(message, 'user');
            input.value = '';

            fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message}) })
                .then(r => r.json()).then(data => addChatMessage(data.response, 'bot'))
                .catch(e => { console.error('Error:', e); addChatMessage('Sorry, I encountered an error processing your request.', 'bot'); });
        }

        function addChatMessage(message, sender) {
            const chatMessages = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = message;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function handleChatKeyPress(event) {
            if (event.key === 'Enter') sendChatMessage();
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/find_path', methods=['POST'])
def find_path():
    try:
        data = request.json
        path, distance = bot.uniform_cost_search(data['start'], data['end'])
        
        if path:
            return jsonify({
                'path': path,
                'distance': distance,
                'walking_time': bot.estimate_walking_time(distance),
                'nodes_explored': bot.search_stats.get('UCS', {}).get('nodes_explored', 0)
            })
        return jsonify({'path': [], 'distance': 0, 'walking_time': '0 seconds', 'nodes_explored': 0, 'error': 'No path found'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        return jsonify({'response': campus_bot.process_query(request.json['message'])})
    except:
        return jsonify({'response': 'Sorry, I encountered an error processing your request.'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'campus_locations': len(bot.campus.locations), 'algorithm': 'UCS (Uniform Cost Search)'})

if __name__ == '__main__':
    print("Starting BotBrain Campus Navigator...\nOpen your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)