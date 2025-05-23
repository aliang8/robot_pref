from flask import Flask, request, jsonify, send_from_directory
import json
import os

app = Flask(__name__, static_url_path='')

# Store preferences
@app.route('/api/preferences', methods=['POST'])
def save_preferences():
    data = request.json
    # Save to file
    with open('preferences.json', 'a') as f:
        f.write(json.dumps(data) + '\n')
    return jsonify({"success": True})

# Serve static files
@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('', path)

@app.route('/')
def root():
    return send_from_directory('', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)