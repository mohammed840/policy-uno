"""
Web server for Uno game.
Flask application with Socket.IO for real-time game updates.
"""

import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass  # dotenv not installed, rely on system env vars

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.game_logic import GameEngine

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'uno-game-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active game sessions
games = {}


def get_game(sid):
    """Get or create game for session."""
    return games.get(sid)


@app.route('/')
def index():
    """Serve main page."""
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    emit('connected', {'sid': request.sid})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    sid = request.sid
    if sid in games:
        del games[sid]
    print(f"Client disconnected: {sid}")


@socketio.on('start_game')
def handle_start_game(data):
    """Start a new game."""
    sid = request.sid
    mode = data.get('mode', 'rl')
    model_key = data.get('model_key')
    
    def on_update():
        """Callback when game state changes."""
        game = games.get(sid)
        if game:
            state = game.get_state()
            socketio.emit('game_state', state, room=sid)
    
    def on_log(message, style='white'):
        """Callback for game logs."""
        socketio.emit('game_log', {'message': message, 'style': style}, room=sid)
    
    try:
        # Create new game engine
        game = GameEngine(
            mode=mode,
            model_key=model_key,
            on_update=on_update,
            on_log=on_log
        )
        games[sid] = game
        
        # Start the game
        game.start_game()
        
        # Send initial state
        state = game.get_state()
        emit('game_started', state)
        
    except Exception as e:
        emit('error', {'message': str(e)})


@socketio.on('play_card')
def handle_play_card(data):
    """Handle card play."""
    sid = request.sid
    game = get_game(sid)
    
    if not game:
        emit('error', {'message': 'No active game'})
        return
    
    action_id = data.get('action_id')
    if action_id is None:
        emit('error', {'message': 'No action specified'})
        return
    
    try:
        game.play_action(action_id)
        state = game.get_state()
        emit('game_state', state)
    except Exception as e:
        emit('error', {'message': str(e)})


@socketio.on('draw_card')
def handle_draw_card():
    """Handle draw card action."""
    sid = request.sid
    game = get_game(sid)
    
    if not game:
        emit('error', {'message': 'No active game'})
        return
    
    try:
        game.play_action(60)  # Draw action
        state = game.get_state()
        emit('game_state', state)
    except Exception as e:
        emit('error', {'message': str(e)})


if __name__ == '__main__':
    print("Starting Uno Web Server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
