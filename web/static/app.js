/**
 * Uno Web Game - Frontend JavaScript
 */

// Game state
let socket = null;
let gameState = null;
let pendingWildAction = null;

// Color and type mappings (RLCard order: Red, Green, Blue, Yellow)
const COLOR_NAMES = ['Red', 'Green', 'Blue', 'Yellow'];
const TYPE_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'Skip', 'Reverse', '+2', 'Wild', 'Wild+4'];

// Initialize socket connection
function initSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });

    socket.on('game_started', (state) => {
        console.log('Game started:', state);
        gameState = state;
        showGameScreen();
        updateGameDisplay();
    });

    socket.on('game_state', (state) => {
        console.log('Game state update:', state);
        gameState = state;
        updateGameDisplay();
    });

    socket.on('game_log', (data) => {
        addLogEntry(data.message, data.style);
    });

    socket.on('error', (data) => {
        console.error('Game error:', data.message);
        addLogEntry('Error: ' + data.message, 'red');
    });
}

// Screen navigation
function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
}

function showMenu() {
    showScreen('menu-screen');
}

function showModelPicker() {
    showScreen('model-screen');
}

function showGameScreen() {
    showScreen('game-screen');
}

function toggleSpectatorMenu() {
    const submenu = document.getElementById('spectator-submenu');
    submenu.classList.toggle('hidden');
}

// Game actions
function startGame(mode, modelKey = null) {
    if (!socket) {
        initSocket();
    }

    // Clear previous logs
    document.getElementById('game-log').innerHTML = '';

    socket.emit('start_game', { mode: mode, model_key: modelKey });
    addLogEntry(`Starting ${mode} mode...`, 'cyan');
}

function playCard(index) {
    if (!gameState || !gameState.is_player_turn) {
        addLogEntry("Not your turn!", 'yellow');
        return;
    }

    if (index >= gameState.player_hand.length) {
        addLogEntry("Invalid card index", 'red');
        return;
    }

    const actionId = gameState.player_hand[index];
    const legalMask = gameState.legal_mask || [];

    // Check if card is playable
    if (legalMask[actionId] <= 0) {
        addLogEntry("That card can't be played right now", 'yellow');
        return;
    }

    // Check if Wild card (needs color selection)
    const typeIdx = actionId % 15;
    if (typeIdx >= 13) {
        pendingWildAction = actionId;
        showColorPicker();
        return;
    }

    socket.emit('play_card', { action_id: actionId });
}

function drawCard() {
    if (!gameState || !gameState.is_player_turn) {
        addLogEntry("Not your turn!", 'yellow');
        return;
    }

    socket.emit('draw_card');
}

function quitGame() {
    showMenu();
}

// Color picker for Wild cards
function showColorPicker() {
    document.getElementById('color-modal').classList.add('active');
}

function hideColorPicker() {
    document.getElementById('color-modal').classList.remove('active');
}

function selectColor(color) {
    hideColorPicker();

    if (pendingWildAction === null) return;

    const colorMap = { 'red': 0, 'yellow': 1, 'green': 2, 'blue': 3 };
    const colorIdx = colorMap[color] || 0;
    const typeIdx = pendingWildAction % 15;

    const newAction = colorIdx * 15 + typeIdx;
    socket.emit('play_card', { action_id: newAction });

    pendingWildAction = null;
}

function cancelColorPicker() {
    hideColorPicker();
    pendingWildAction = null;
}

// Update game display
function updateGameDisplay() {
    if (!gameState) return;

    updateOpponentDisplay();
    updateDiscardPile();
    updatePlayerHand();
    updateStatus();
    updateRLDecision();
}

// Toggle RL Panel visibility
function toggleRLPanel() {
    const panel = document.getElementById('rl-panel');
    panel.classList.toggle('collapsed');
}

function updateOpponentDisplay() {
    const nameEl = document.getElementById('opponent-name');
    const cardsEl = document.getElementById('opponent-cards');
    const turnEl = document.getElementById('opponent-turn-indicator');

    nameEl.textContent = gameState.opponent_name || 'Opponent';
    cardsEl.textContent = `Cards: ${gameState.opponent_cards || 0}`;

    if (!gameState.is_player_turn && !gameState.game_over) {
        turnEl.classList.add('active');
    } else {
        turnEl.classList.remove('active');
    }
}

function updateDiscardPile() {
    const cardEl = document.getElementById('top-card');
    const colorEl = document.getElementById('active-color');

    const topCard = gameState.top_card_action;
    if (topCard === null || topCard === undefined) {
        cardEl.className = 'card-display';
        cardEl.innerHTML = '<span class="card-value">?</span>';
        colorEl.textContent = 'Active: -';
        return;
    }

    const colorIdx = Math.floor(topCard / 15);
    const typeIdx = topCard % 15;

    const colorClass = getColorClass(topCard);
    const cardText = TYPE_NAMES[typeIdx] || '?';

    cardEl.className = `card-display ${colorClass}`;
    cardEl.innerHTML = `<span class="card-value">${cardText}</span>`;

    colorEl.textContent = `Active: ${gameState.active_color || COLOR_NAMES[colorIdx]}`;
}

function updatePlayerHand() {
    const container = document.getElementById('player-hand');
    container.innerHTML = '';

    const hand = gameState.player_hand || [];
    const legalMask = gameState.legal_mask || [];

    hand.forEach((actionId, index) => {
        const isPlayable = legalMask[actionId] > 0;
        const btn = createCardButton(actionId, index, isPlayable);
        container.appendChild(btn);
    });
}

function createCardButton(actionId, index, isPlayable) {
    const btn = document.createElement('button');
    const colorClass = getColorClass(actionId);
    const cardText = actionToCardString(actionId);

    btn.className = `card-button ${colorClass}${isPlayable ? '' : ' disabled'}`;
    btn.textContent = `[${index}] ${cardText}`;
    btn.onclick = () => playCard(index);

    return btn;
}

function updateStatus() {
    const statusEl = document.getElementById('status-panel');

    if (gameState.game_over) {
        if (gameState.is_spectator) {
            statusEl.textContent = `🎉 ${gameState.winner} Won! 🎉`;
            statusEl.className = 'status-panel';
        } else if (gameState.winner === 'player') {
            statusEl.textContent = '🎉 You Won! 🎉';
            statusEl.className = 'status-panel player-turn';
        } else {
            statusEl.textContent = 'Game Over - Opponent Won';
            statusEl.className = 'status-panel';
        }
    } else if (gameState.is_spectator) {
        // Spectator mode - show which AI's turn it is
        statusEl.textContent = `${gameState.current_player_name}'s Turn...`;
        statusEl.className = 'status-panel';
    } else if (gameState.is_player_turn) {
        statusEl.textContent = 'Your Turn - Select a card';
        statusEl.className = 'status-panel player-turn';
    } else {
        statusEl.textContent = "Opponent's Turn...";
        statusEl.className = 'status-panel';
    }
}

// Helper functions
function actionToCardString(actionId) {
    if (actionId === 60) return 'Draw';

    const colorIdx = Math.floor(actionId / 15);
    const typeIdx = actionId % 15;

    if (typeIdx >= 13) {
        return `${TYPE_NAMES[typeIdx]}`;
    }
    return `${COLOR_NAMES[colorIdx]} ${TYPE_NAMES[typeIdx]}`;
}

function getColorClass(actionId) {
    if (actionId === 60) return '';

    const colorIdx = Math.floor(actionId / 15);
    const typeIdx = actionId % 15;

    // RLCard order: Red, Green, Blue, Yellow
    if (typeIdx >= 13) return 'wild';
    return ['red', 'green', 'blue', 'yellow'][colorIdx];
}

// Update RL Decision Panel
function updateRLDecision() {
    const contentEl = document.getElementById('rl-decision-content');
    if (!contentEl) return;

    const decision = gameState?.rl_decision;

    if (!decision) {
        contentEl.innerHTML = '<p class="waiting-text">Waiting for RL agent\'s turn...</p>';
        return;
    }

    // Build Q-value visualization
    let html = `
        <div class="decision-summary">
            <div class="selected-action">
                <span class="label">Selected:</span>
                <span class="value card-tag ${getColorClass(decision.selected_action)}">${decision.selected_card}</span>
            </div>
            <div class="q-value">
                <span class="label">Q-value:</span>
                <span class="value">${decision.selected_q_value.toFixed(4)}</span>
            </div>
            <div class="confidence">
                <span class="label">Confidence:</span>
                <span class="value">${(decision.confidence * 100).toFixed(1)}%</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${decision.confidence * 100}%"></div>
                </div>
            </div>
        </div>
        
        <div class="action-ranking">
            <h5>Action Ranking by Q(s,a)</h5>
            <div class="ranking-list">
    `;

    // Add top actions
    if (decision.top_actions) {
        decision.top_actions.forEach((action, idx) => {
            const isSelected = action.is_selected ? 'selected' : '';
            const colorClass = getColorClass(action.action_id);
            html += `
                <div class="rank-item ${isSelected}">
                    <span class="rank">#${idx + 1}</span>
                    <span class="card-tag ${colorClass}">${action.card_name}</span>
                    <span class="q-val">${action.q_value.toFixed(3)}</span>
                </div>
            `;
        });
    }

    html += `
            </div>
        </div>
        
        <div class="decision-stats">
            <span class="stat">Legal actions: ${decision.num_legal_actions}</span>
            <span class="stat">Q range: [${decision.q_value_range?.min?.toFixed(2)}, ${decision.q_value_range?.max?.toFixed(2)}]</span>
        </div>
    `;

    contentEl.innerHTML = html;
}

function addLogEntry(message, style = 'white') {
    const logEl = document.getElementById('game-log');
    const entry = document.createElement('div');
    entry.className = `log-entry ${style}`;
    entry.textContent = message;
    logEl.appendChild(entry);
    logEl.scrollTop = logEl.scrollHeight;
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    const gameScreen = document.getElementById('game-screen');
    if (!gameScreen.classList.contains('active')) return;

    const key = e.key.toLowerCase();

    if (key >= '0' && key <= '9') {
        playCard(parseInt(key));
        e.preventDefault();
    } else if (key === 'd') {
        drawCard();
        e.preventDefault();
    } else if (key === 'q') {
        quitGame();
        e.preventDefault();
    } else if (key === 'escape') {
        cancelColorPicker();
    } else if (key === 'r') {
        selectColor('red');
    } else if (key === 'y') {
        selectColor('yellow');
    } else if (key === 'g') {
        selectColor('green');
    } else if (key === 'b') {
        selectColor('blue');
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initSocket();
});
