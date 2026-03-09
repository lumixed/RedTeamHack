/**
 * Find My Force — COP Dashboard Frontend
 * Real-time RF track visualization with Leaflet + Socket.IO
 */

// ── CONFIG ─────────────────────────────────────────────────────────────────
const AFFILIATION_COLORS = {
    friendly: '#00ffd0', // Neon Cyan/Green
    hostile: '#ff2d55', // Neon Red
    unknown: '#ff9f0a', // Neon Orange
    civilian: '#5856d6', // Tactical Indigo
    stale: '#8e8e93', // System Gray
};

const LABEL_COLORS = {
    'Radar-Altimeter': '#00ffd0',
    'Satcom': '#00ffd0',
    'short-range': '#00ffd0',
    'Airborne-detection': '#ff2d55',
    'Airborne-range': '#ff2d55',
    'Air-Ground-MTI': '#ff2d55',
    'EW-Jammer': '#ff2d55',
    'AM radio': '#5856d6',
    'unknown': '#ff9f0a',
};

// ── STATE ──────────────────────────────────────────────────────────────────
let map = null;
let socket = null;
let allTracks = {};          // track_id → track dict
let trackMarkers = {};       // track_id → Leaflet marker
let uncertaintyCircles = {}; // track_id → Leaflet circle
let pathPolylines = {};      // track_id → Leaflet polyline
let receiverMarkers = [];    // Leaflet markers for receivers
let selectedTrackId = null;
let activeFilter = 'all';
let searchQuery = '';
let signalFilter = 'all';
let confidenceFilter = 0;
let showTrails = true;
let showProjections = true;
let maxObsFeed = 50;
let projectionPolylines = {}; // track_id → Leaflet polyline


// ── INIT MAP ───────────────────────────────────────────────────────────────
function initMap() {
    map = L.map('map', {
        center: [49.26, -123.25],
        zoom: 13,
        zoomControl: true,
        attributionControl: false,
    });

    // Dark OSM tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap',
        opacity: 0.9,
    }).addTo(map);

    // Update coords display on move
    map.on('mousemove', (e) => {
        document.getElementById('map-coords').textContent =
            `${e.latlng.lat.toFixed(4)}°N ${Math.abs(e.latlng.lng).toFixed(4)}°W`;
    });

    console.log('[MAP] Initialized');
}

// ── SOCKET.IO ──────────────────────────────────────────────────────────────
function initSocket() {
    socket = io({ transports: ['websocket', 'polling'] });

    socket.on('connect', () => {
        console.log('[WS] Connected:', socket.id);
        updateSimState('CONNECTED', true);
        socket.emit('request_tracks');
    });

    socket.on('disconnect', () => {
        console.log('[WS] Disconnected');
        updateSimState('DISCONNECTED', false);
    });

    socket.on('init', (data) => {
        console.log('[WS] Init received');
        if (data.tracks) updateAllTracks(data.tracks);
        if (data.status) updateSystemStatus(data.status);
        if (data.receivers) renderReceivers(data.receivers.receivers || []);
        if (data.stats) updateTrackStats(data.stats);
    });

    socket.on('track_update', (data) => {
        if (data.all_tracks) updateAllTracks(data.all_tracks);
        if (data.stats) updateTrackStats(data.stats);
    });

    socket.on('tracks_broadcast', (data) => {
        if (data.tracks) updateAllTracks(data.tracks);
        if (data.stats) updateTrackStats(data.stats);
        if (data.feed_stats) updateFeedStats(data.feed_stats);
    });

    socket.on('observation', (obs) => {
        addObservationToFeed(obs);
    });

    socket.on('training_complete', (data) => {
        toast(`Training complete! F1: ${(data.metrics?.f1_macro * 100).toFixed(1)}%`, 'success');
        document.getElementById('training-progress').style.display = 'none';
        document.getElementById('cls-status').textContent = 'TRAINED';
        document.getElementById('cls-status').className = 'status-val online';
        document.getElementById('btn-train').disabled = false;
    });

    socket.on('training_error', (data) => {
        toast(`Training error: ${data.error}`, 'error');
        document.getElementById('training-progress').style.display = 'none';
        document.getElementById('btn-train').disabled = false;
    });

    socket.on('eval_complete', (data) => {
        if (data.result) {
            toast('Evaluation submitted! Refreshing score...', 'success');
            setTimeout(loadScore, 2000);
        } else {
            toast('Eval not yet open or submission failed', 'warning');
        }
        document.getElementById('btn-eval').disabled = false;
        document.getElementById('btn-eval').textContent = 'SUBMIT EVAL';
    });

    socket.on('eval_started', () => {
        toast('Running evaluation submission...', 'warning');
    });
}

// ── TRACK RENDERING ────────────────────────────────────────────────────────
function updateAllTracks(tracks) {
    allTracks = {};
    tracks.forEach(t => { allTracks[t.track_id] = t; });

    // Update/create markers
    tracks.forEach(track => renderTrackMarker(track));

    // Remove markers for gone tracks
    Object.keys(trackMarkers).forEach(id => {
        if (!allTracks[id]) removeTrackMarker(id);
    });

    // Update track list panel
    renderTrackList();

    // Update detail panel if selected track changed
    if (selectedTrackId && allTracks[selectedTrackId]) {
        renderTrackDetail(allTracks[selectedTrackId]);
    }

    // Update map counter
    const active = tracks.filter(t => t.state !== 'lost');
    document.getElementById('track-counter-map').textContent = `TRACKS: ${active.length}`;
}

function renderTrackMarker(track) {
    const isStale = track.is_stale;
    const affiliation = track.affiliation;
    const color = isStale ? AFFILIATION_COLORS.stale : (AFFILIATION_COLORS[affiliation] || AFFILIATION_COLORS.unknown);
    const lat = track.latitude;
    const lon = track.longitude;

    if (!lat || !lon) return;

    // Create or update marker
    if (trackMarkers[track.track_id]) {
        trackMarkers[track.track_id].setLatLng([lat, lon]);
    } else {
        const icon = createTrackIcon(track, color);
        const marker = L.marker([lat, lon], { icon, zIndexOffset: affiliation === 'hostile' ? 1000 : 500 });
        marker.on('click', () => selectTrack(track.track_id));
        marker.addTo(map);
        trackMarkers[track.track_id] = marker;
    }

    // Update icon appearance
    updateTrackIcon(track.track_id, track, color);

    // Uncertainty circle
    if (uncertaintyCircles[track.track_id]) {
        uncertaintyCircles[track.track_id].setLatLng([lat, lon]);
        uncertaintyCircles[track.track_id].setRadius(track.uncertainty_m || 200);
        uncertaintyCircles[track.track_id].setStyle({ color: color, fillColor: color });
    } else {
        const circle = L.circle([lat, lon], {
            radius: track.uncertainty_m || 200,
            color: color,
            fillColor: color,
            fillOpacity: 0.05,
            weight: 1,
            dashArray: '4 4',
            opacity: 0.4,
            className: 'uncertainty-circle',
        });
        circle.addTo(map);
        uncertaintyCircles[track.track_id] = circle;
    }

    // --- THE TRAIL (Historical Paths) ---
    const historyPoints = (track.position_history || []).map(p => [p.latitude, p.longitude]);
    if (historyPoints.length > 1) {
        if (pathPolylines[track.track_id]) {
            pathPolylines[track.track_id].setLatLngs(historyPoints);
            pathPolylines[track.track_id].setStyle({ color: color });
        } else {
            const poly = L.polyline(historyPoints, {
                color: color,
                weight: 2,
                opacity: 0.4,
                dashArray: '2, 6',
                smoothFactor: 1
            }).addTo(map);
            pathPolylines[track.track_id] = poly;
        }
    }

    // --- AI PROJECTION (Predicted Paths) ---
    if (track.velocity_mps && track.velocity_mps.speed_mps > 0.1) {
        const projectedPoints = calculateProjection(track);
        if (projectedPoints.length > 1) {
            if (projectionPolylines[track.track_id]) {
                projectionPolylines[track.track_id].setLatLngs(projectedPoints);
                projectionPolylines[track.track_id].setStyle({ color: color });
            } else {
                const poly = L.polyline(projectedPoints, {
                    color: color,
                    weight: 2,
                    opacity: 0.6,
                    dashArray: '5, 10',
                    className: 'projection-line'
                }).addTo(map);
                projectionPolylines[track.track_id] = poly;
            }
        }
    } else if (projectionPolylines[track.track_id]) {
        map.removeLayer(projectionPolylines[track.track_id]);
        delete projectionPolylines[track.track_id];
    }

    // Bind popup
    const marker = trackMarkers[track.track_id];
    marker.unbindPopup();
    marker.bindPopup(buildPopupContent(track), { maxWidth: 260 });
}

function createTrackIcon(track, color) {
    const html = `
    <div class="track-marker" title="${track.track_id}">
      <div class="marker-outer" style="border-color:${color};background:${color}18">
        <div class="marker-inner" style="background:${color}"></div>
        <div class="marker-ring" style="border-color:${color}"></div>
      </div>
    </div>`;

    return L.divIcon({
        html,
        className: '',
        iconSize: [36, 36],
        iconAnchor: [18, 28],
    });
}

function updateTrackIcon(trackId, track, color) {
    const marker = trackMarkers[trackId];
    if (!marker) return;
    const icon = createTrackIcon(track, color);
    marker.setIcon(icon);
}

function getTrackShape(affiliation) {
    const shapes = { friendly: '▲', hostile: '◆', unknown: '●', civilian: '■' };
    return shapes[affiliation] || '●';
}

function removeTrackMarker(trackId) {
    if (trackMarkers[trackId]) { map.removeLayer(trackMarkers[trackId]); delete trackMarkers[trackId]; }
    if (uncertaintyCircles[trackId]) { map.removeLayer(uncertaintyCircles[trackId]); delete uncertaintyCircles[trackId]; }
    if (pathPolylines[trackId]) { map.removeLayer(pathPolylines[trackId]); delete pathPolylines[trackId]; }
    if (projectionPolylines[trackId]) { map.removeLayer(projectionPolylines[trackId]); delete projectionPolylines[trackId]; }
}

function calculateProjection(track) {
    const points = [[track.latitude, track.longitude]];
    const vel = track.velocity_mps;
    if (!vel) return points;

    // Meters to degrees (rough approximation for projection)
    const latPerMeter = 1 / 111111;
    const lonPerMeter = 1 / (111111 * Math.cos(track.latitude * Math.PI / 180));

    // Project 30, 60, 90 seconds ahead
    [30, 60, 90].forEach(seconds => {
        const nextLat = track.latitude + (vel.vy_mps * seconds * latPerMeter);
        const nextLon = track.longitude + (vel.vx_mps * seconds * lonPerMeter);
        points.push([nextLat, nextLon]);
    });
    return points;
}

function buildPopupContent(track) {
    const affilColor = AFFILIATION_COLORS[track.affiliation] || '#fff';
    const age = track.age_seconds ? `${Math.round(track.age_seconds)}s ago` : 'now';
    const conf = (track.classification_confidence * 100).toFixed(0);
    const lat = track.latitude?.toFixed(5) ?? '–';
    const lon = track.longitude?.toFixed(5) ?? '–';
    const unc = track.uncertainty_m ? `±${Math.round(track.uncertainty_m)}m` : '–';

    return `
    <div class="popup-content">
      <div class="popup-title" style="color:${affilColor}">${track.track_id}</div>
      <div class="popup-row"><span class="popup-label">SIGNAL TYPE</span><span class="popup-val">${track.classification_label?.toUpperCase()}</span></div>
      <div class="popup-row"><span class="popup-label">AFFILIATION</span><span class="popup-val" style="color:${affilColor}">${track.affiliation?.toUpperCase()}</span></div>
      <div class="popup-row"><span class="popup-label">CONFIDENCE</span><span class="popup-val">${conf}%</span></div>
      <div class="popup-row"><span class="popup-label">POSITION</span><span class="popup-val">${lat}, ${lon}</span></div>
      <div class="popup-row"><span class="popup-label">ACCURACY</span><span class="popup-val">${unc}</span></div>
      <div class="popup-row"><span class="popup-label">METHOD</span><span class="popup-val">${track.geolocation_method?.toUpperCase() ?? '–'}</span></div>
      <div class="popup-row"><span class="popup-label">RECEIVERS</span><span class="popup-val">${track.n_receivers}</span></div>
      <div class="popup-row"><span class="popup-label">STATE</span><span class="popup-val">${track.state?.toUpperCase()}</span></div>
      <div class="popup-row"><span class="popup-label">LAST SEEN</span><span class="popup-val">${age}</span></div>
      <div class="popup-row"><span class="popup-label">OBSERVATIONS</span><span class="popup-val">${track.observation_count}</span></div>
    </div>`;
}

// ── TRACK LIST PANEL ────────────────────────────────────────────────────────
function renderTrackList() {
    const list = document.getElementById('track-list');
    const tracks = Object.values(allTracks)
        .filter(t => t.state !== 'lost')
        .filter(t => activeFilter === 'all' || t.affiliation === activeFilter)
        .filter(t => !searchQuery || t.track_id.toLowerCase().includes(searchQuery.toLowerCase()))
        .filter(t => signalFilter === 'all' || t.classification_label === signalFilter)
        .filter(t => (t.classification_confidence * 100) >= confidenceFilter)
        .sort((a, b) => b.last_seen - a.last_seen);

    // Also update map filters
    Object.keys(allTracks).forEach(id => {
        const track = allTracks[id];
        const marker = trackMarkers[id];
        const circle = uncertaintyCircles[id];
        const poly = pathPolylines[id];
        const proj = projectionPolylines[id];

        const visible = tracks.some(t => t.track_id === id);

        if (marker) visible ? marker.addTo(map) : map.removeLayer(marker);
        if (circle) visible ? circle.addTo(map) : map.removeLayer(circle);
        if (poly) (visible && showTrails) ? poly.addTo(map) : map.removeLayer(poly);
        if (proj) (visible && showProjections) ? proj.addTo(map) : map.removeLayer(proj);
    });

    document.getElementById('track-count-badge').textContent = tracks.length;

    if (tracks.length === 0) {
        list.innerHTML = `
      <div class="empty-state">
        <svg viewBox="0 0 48 48" fill="none" class="empty-icon">
          <circle cx="24" cy="24" r="20" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 4"/>
          <circle cx="24" cy="24" r="8" stroke="currentColor" stroke-width="1.5"/>
          <circle cx="24" cy="24" r="2" fill="currentColor"/>
        </svg>
        <span>No tracks match filter</span>
      </div>`;
        return;
    }

    list.innerHTML = tracks.map(t => buildTrackCard(t)).join('');
}

function buildTrackCard(track) {
    const isSelected = track.track_id === selectedTrackId;
    const isStale = track.is_stale;
    const affil = track.affiliation || 'unknown';
    const color = AFFILIATION_COLORS[affil] || '#fff';
    const conf = (track.classification_confidence * 100).toFixed(0);
    const age = track.age_seconds < 60
        ? `${Math.round(track.age_seconds)}s`
        : `${Math.round(track.age_seconds / 60)}m`;

    const stateClass = {
        'confirmed': 'confirmed',
        'tentative': 'tentative',
        'coasting': 'coasting',
    }[track.state] || '';

    return `
    <div class="track-card ${affil} ${isStale ? 'stale' : ''} ${isSelected ? 'selected' : ''}"
         onclick="selectTrack('${track.track_id}')" id="card-${track.track_id}">
      <div class="track-card-header">
        <span class="track-id">${track.track_id}</span>
        <span class="track-state-tag ${stateClass}">${track.state?.toUpperCase()}</span>
      </div>
      <div class="track-label ${affil}">${track.classification_label?.toUpperCase()}</div>
      <div class="track-meta">
        <div class="track-conf-bar">
          <div class="track-conf-fill ${affil}" style="width:${conf}%"></div>
        </div>
        <span>${conf}%</span>
        <span style="color:${color}33">|</span>
        <span>${age} ago</span>
        <span style="color:var(--text-dim)">RX:${track.n_receivers}</span>
      </div>
    </div>`;
}

// ── TRACK DETAIL ────────────────────────────────────────────────────────────
function selectTrack(trackId) {
    selectedTrackId = trackId;
    const track = allTracks[trackId];
    if (!track) return;

    // Update list selection
    document.querySelectorAll('.track-card').forEach(c => c.classList.remove('selected'));
    const card = document.getElementById(`card-${trackId}`);
    if (card) card.classList.add('selected');

    // Pan map to track
    if (track.latitude && track.longitude) {
        map.panTo([track.latitude, track.longitude]);
        const marker = trackMarkers[trackId];
        if (marker) marker.openPopup();
    }

    renderTrackDetail(track);
    document.getElementById('detail-close').style.display = 'block';
}

function clearSelection() {
    selectedTrackId = null;
    document.querySelectorAll('.track-card').forEach(c => c.classList.remove('selected'));
    document.getElementById('track-detail').innerHTML = `
    <div class="track-detail-empty">
      <svg viewBox="0 0 48 48" fill="none" class="empty-icon">
        <rect x="8" y="8" width="32" height="32" rx="4" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 4"/>
        <path d="M16 20H32M16 24H28M16 28H24" stroke="currentColor" stroke-width="1.5"/>
      </svg>
      <span>Select a track for details</span>
    </div>`;
    document.getElementById('detail-close').style.display = 'none';
}

function renderTrackDetail(track) {
    const affil = track.affiliation || 'unknown';
    const color = AFFILIATION_COLORS[affil] || '#fff';
    const conf = (track.classification_confidence * 100).toFixed(1);
    const age = track.age_seconds < 60
        ? `${Math.round(track.age_seconds)}s ago`
        : `${Math.round(track.age_seconds / 60)}m ago`;

    const vel = track.velocity_mps;
    const velStr = vel ? `${vel.speed_mps.toFixed(1)} m/s` : 'STATIONARY';

    document.getElementById('track-detail').innerHTML = `
    <div class="detail-content">
      <div class="detail-affil ${affil}">
        <span style="width:8px;height:8px;border-radius:50%;background:${color};display:inline-block"></span>
        ${affil.toUpperCase()}
      </div>
      <div class="detail-label">${track.classification_label?.toUpperCase()}</div>

      <div class="detail-row">
        <span class="detail-row-label">TRACK ID</span>
        <span class="detail-row-val">${track.track_id}</span>
      </div>
      <div class="detail-row">
        <span class="detail-row-label">STATE</span>
        <span class="detail-row-val">${track.state?.toUpperCase()}</span>
      </div>
      <div class="detail-row">
        <span class="detail-row-label">CONFIDENCE</span>
        <div class="confidence-display">
          <div class="conf-bar">
            <div class="conf-bar-fill" style="width:${conf}%;background:${color}"></div>
          </div>
          <span class="detail-row-val">${conf}%</span>
        </div>
      </div>
      <div class="detail-row">
        <span class="detail-row-label">POSITION</span>
        <span class="detail-row-val">${track.latitude?.toFixed(5)}, ${track.longitude?.toFixed(5)}</span>
      </div>
      <div class="detail-row">
        <span class="detail-row-label">ACCURACY</span>
        <span class="detail-row-val">±${Math.round(track.uncertainty_m ?? 0)}m</span>
      </div>
      <div class="detail-row">
        <span class="detail-row-label">GEO METHOD</span>
        <span class="detail-row-val">${track.geolocation_method?.toUpperCase() ?? '–'}</span>
      </div>
      <div class="detail-row">
        <span class="detail-row-label">RECEIVERS</span>
        <span class="detail-row-val">${track.n_receivers}</span>
      </div>
      <div class="detail-row">
        <span class="detail-row-label">VELOCITY</span>
        <span class="detail-row-val">${velStr}</span>
      </div>
      <div class="detail-row">
        <span class="detail-row-label">LAST SEEN</span>
        <span class="detail-row-val">${age}</span>
      </div>
      <div class="detail-row">
        <span class="detail-row-label">OBSERVATIONS</span>
        <span class="detail-row-val">${track.observation_count}</span>
      </div>
      <div class="detail-row">
        <span class="detail-row-label">STALE</span>
        <span class="detail-row-val" style="color:${track.is_stale ? 'var(--col-danger)' : 'var(--col-success)'}">
          ${track.is_stale ? 'YES' : 'NO'}
        </span>
      </div>
    </div>`;
}

// ── RECEIVERS ───────────────────────────────────────────────────────────────
function renderReceivers(receivers) {
    receiverMarkers.forEach(m => map.removeLayer(m));
    receiverMarkers = [];

    receivers.forEach(rx => {
        const icon = L.divIcon({
            html: `
        <div style="
          width:14px; height:14px;
          background:rgba(0,145,234,0.15);
          border:2px solid #0091ea;
          border-radius:50%;
          display:flex; align-items:center; justify-content:center;
        ">
          <div style="width:4px;height:4px;background:#0091ea;border-radius:50%"></div>
        </div>`,
            className: '',
            iconSize: [14, 14],
            iconAnchor: [7, 7],
        });

        const marker = L.marker([rx.latitude, rx.longitude], { icon });
        marker.bindTooltip(`
      <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#e8f4f0">
        <b style="color:#0091ea">${rx.receiver_id}</b><br>
        Sens: ${rx.sensitivity_dbm} dBm<br>
        ToA acc: ${rx.timing_accuracy_ns} ns
      </div>`, {
            permanent: false,
            className: 'leaflet-tooltip',
            direction: 'top',
        });
        marker.addTo(map);
        receiverMarkers.push(marker);
    });

    if (receivers.length > 0) {
        const lats = receivers.map(r => r.latitude);
        const lons = receivers.map(r => r.longitude);
        const centerLat = (Math.min(...lats) + Math.max(...lats)) / 2;
        const centerLon = (Math.min(...lons) + Math.max(...lons)) / 2;
        map.setView([centerLat, centerLon], 13);
    }
}

// ── OBSERVATION FEED ────────────────────────────────────────────────────────
function addObservationToFeed(obs) {
    const feed = document.getElementById('obs-feed');
    const cls = obs.classification;
    const label = cls?.label ?? 'unknown';
    const affil = inferAffiliation(label);
    const conf = cls ? `${(cls.confidence * 100).toFixed(0)}%` : '–';
    const rssi = obs.rssi_dbm ? `${obs.rssi_dbm.toFixed(1)} dBm` : '–';
    const time = new Date().toLocaleTimeString('en-CA', {
        timeZone: 'America/Vancouver',
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    }) + ' PST';

    // Remove empty state if present
    const empty = feed.querySelector('.empty-state');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = 'obs-item';
    item.innerHTML = `
    <div class="obs-item-header">
      <span class="obs-rx">${obs.receiver_id ?? 'RX-?'}</span>
      <span class="obs-time">${time}</span>
    </div>
    <div class="obs-label ${affil}">${label.toUpperCase()} <span style="color:var(--text-dim);font-size:10px">(${conf})</span></div>
    <div class="obs-rssi">RSSI: ${rssi}</div>`;

    feed.insertBefore(item, feed.firstChild);

    // Limit entries
    const items = feed.querySelectorAll('.obs-item');
    if (items.length > maxObsFeed) {
        items[items.length - 1].remove();
    }
}

function inferAffiliation(label) {
    const friendly = ['Radar-Altimeter', 'Satcom', 'short-range'];
    const hostile = ['Airborne-detection', 'Airborne-range', 'Air-Ground-MTI', 'EW-Jammer'];
    const civilian = ['AM radio'];
    if (friendly.includes(label)) return 'friendly';
    if (hostile.includes(label)) return 'hostile';
    if (civilian.includes(label)) return 'civilian';
    return 'unknown';
}

// ── STATUS UPDATES ──────────────────────────────────────────────────────────
function updateTrackStats(stats) {
    document.getElementById('count-friendly').textContent = stats.friendly ?? 0;
    document.getElementById('count-hostile').textContent = stats.hostile ?? 0;
    document.getElementById('count-unknown').textContent = (stats.unknown ?? 0) + (stats.civilian ?? 0);
}

function updateFeedStats(stats) {
    document.getElementById('obs-count').textContent = stats.observations_received ?? 0;
    document.getElementById('submit-count').textContent = stats.submissions_sent ?? 0;

    if (stats.start_time) {
        document.getElementById('feed-status').textContent = 'LIVE';
        document.getElementById('feed-status').className = 'status-val online';
    }
}

function updateSystemStatus(status) {
    if (status.pipeline_running) {
        document.getElementById('pipeline-badge').textContent = 'ONLINE';
        document.getElementById('pipeline-badge').className = 'section-badge green';
    }
    if (status.classifier_trained) {
        document.getElementById('cls-status').textContent = 'TRAINED';
        document.getElementById('cls-status').className = 'status-val online';
    }
    if (status.api_key_set) {
        document.getElementById('feed-status').textContent = 'RUNNING';
        document.getElementById('feed-status').className = 'status-val online';
    }
}

function updateSimState(state, connected) {
    const badge = document.getElementById('sim-state-badge');
    const text = document.getElementById('sim-state-text');
    text.textContent = state;
    const dot = badge.querySelector('.pulse-dot');
    if (connected) {
        dot.style.background = 'var(--col-accent)';
    } else {
        dot.style.background = 'var(--col-danger)';
    }
}

// ── FILTER ──────────────────────────────────────────────────────────────────
function setFilter(btn, filter) {
    activeFilter = filter;
    document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    renderTrackList();
}

function handleFilterChange() {
    searchQuery = document.getElementById('track-search').value;
    signalFilter = document.getElementById('signal-filter').value;
    confidenceFilter = parseInt(document.getElementById('conf-filter').value);
    showTrails = document.getElementById('toggle-trails').checked;
    showProjections = document.getElementById('toggle-projections').checked;
    document.getElementById('conf-val').textContent = confidenceFilter;
    renderTrackList();
}

// ── ACTIONS ─────────────────────────────────────────────────────────────────
async function trainModel() {
    const btn = document.getElementById('btn-train');
    btn.disabled = true;
    document.getElementById('training-progress').style.display = 'block';
    document.getElementById('cls-status').textContent = 'TRAINING';
    document.getElementById('cls-status').className = 'status-val warning';

    try {
        const resp = await fetch('/api/train', { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) {
            toast('Training started in background...', 'success');
        } else {
            toast(`Training error: ${data.error}`, 'error');
            btn.disabled = false;
            document.getElementById('training-progress').style.display = 'none';
        }
    } catch (e) {
        toast(`Network error: ${e.message}`, 'error');
        btn.disabled = false;
        document.getElementById('training-progress').style.display = 'none';
    }
}

async function loadScore() {
    try {
        const resp = await fetch('/api/score/fetch');
        const data = await resp.json();

        if (data.error) {
            toast('Could not fetch score', 'warning');
            return;
        }

        // Show score section
        const section = document.getElementById('score-section');
        section.style.display = 'block';

        const total = data.total_score?.toFixed(1) ?? '–';
        const cls = data.classification_score?.toFixed(1) ?? '–';
        const geo = data.geolocation_score?.toFixed(1) ?? '–';
        const nov = data.novelty_detection_score?.toFixed(1) ?? '–';

        document.getElementById('total-score-badge').textContent = `${total}`;

        document.getElementById('score-breakdown').innerHTML = `
      <div class="score-row">
        <span class="score-row-label">TOTAL</span>
        <span class="score-row-val">${total}</span>
      </div>
      <div class="score-row">
        <div>
          <div style="display:flex;justify-content:space-between"><span class="score-row-label">CLASSIFICATION (40%)</span><span class="score-row-val">${cls}</span></div>
          <div class="score-bar-wrap"><div class="score-bar" style="width:${cls}%"></div></div>
        </div>
      </div>
      <div class="score-row">
        <div>
          <div style="display:flex;justify-content:space-between"><span class="score-row-label">GEOLOCATION (30%)</span><span class="score-row-val">${geo}</span></div>
          <div class="score-bar-wrap"><div class="score-bar" style="width:${geo}%;background:var(--col-accent2)"></div></div>
        </div>
      </div>
      <div class="score-row">
        <div>
          <div style="display:flex;justify-content:space-between"><span class="score-row-label">NOVELTY (30%)</span><span class="score-row-val">${nov}</span></div>
          <div class="score-bar-wrap"><div class="score-bar" style="width:${nov}%;background:var(--col-warning)"></div></div>
        </div>
      </div>
      <div class="score-row">
        <span class="score-row-label">SUBMISSIONS</span>
        <span class="score-row-val">${data.submissions_count ?? 0}</span>
      </div>
      <div class="score-row">
        <span class="score-row-label">TEAM</span>
        <span class="score-row-val">${data.team_name ?? '–'}</span>
      </div>`;

        toast(`Score: ${total} (Cls: ${cls}, Geo: ${geo}, Nov: ${nov})`, 'success');
    } catch (e) {
        toast(`Error: ${e.message}`, 'error');
    }
}

async function runEval() {
    const btn = document.getElementById('btn-eval');
    btn.disabled = true;
    btn.textContent = 'SUBMITTING...';
    socket.emit('request_eval');
}

// ── CLOCK ───────────────────────────────────────────────────────────────────
function updateClock() {
    const now = new Date();

    // Format the time in America/Vancouver timezone
    const timeStr = now.toLocaleTimeString('en-CA', {
        timeZone: 'America/Vancouver',
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });

    document.getElementById('clock').textContent = timeStr + ' PST';
}

// ── TOAST ────────────────────────────────────────────────────────────────────
function toast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), 4200);
}

// ── PERIODIC STATUS POLL ─────────────────────────────────────────────────────
async function pollStatus() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        if (data.system) updateSystemStatus(data.system);
        if (data.tracks) updateTrackStats(data.tracks);
        if (data.feed_stats) updateFeedStats(data.feed_stats);
        if (data.server_health) {
            const simState = data.server_health.simulation_state?.toUpperCase() ?? 'UNKNOWN';
            updateSimState(simState, true);
        }
    } catch (e) {
        // Ignore
    }
}

// ── BOOTSTRAP ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initSocket();

    setInterval(updateClock, 1000);
    updateClock();

    setInterval(pollStatus, 10000);
    pollStatus();

    map.on('click', () => {
        if (!selectedTrackId) {
            // Map click outside track - clear selection
        }
    });

    console.log('[APP] Find My Force COP initialized');
});
