from main import handle_command
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/command', methods=['POST'])
def command():
    data = request.get_json()
    response = handle_command(data['command'])
    return jsonify(response)