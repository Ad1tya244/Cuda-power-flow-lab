/**
 * index.js - Cloud Compute Server & WebSocket API Gateway
 * Zero-dependency native Node.js HTTP & RFC 6455 WebSocket Server
 * Connects Browser Frontend to the Remote Cloud GPU Power Flow Solver.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');
const crypto = require('crypto');

const {
  getIEEE14Case,
  getIEEE30Case,
  generateSyntheticGrid,
  parseCaseFile,
  serializeCaseFile
} = require('./cases');
const { solvePowerFlow } = require('./solver');
const { runNMinus1Analysis } = require('./contingency');
const {
  getGpuStatus,
  setGpuProfile,
  setRemoteEndpoint,
  loadBenchmarkData,
  executeRemoteCudaPowerFlow
} = require('./cuda_bridge');

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = path.join(__dirname, '../public');

// MIME types for static assets
const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon'
};

// WebSocket connection registry
const wsClients = new Set();

/**
 * Handles HTTP requests (Static files & REST APIs)
 */
const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;
  const method = req.method;

  // CORS headers for flexible cloud deployment
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  // --- REST API ENDPOINTS ---
  if (pathname.startsWith('/api/')) {
    // Helper to send JSON
    const sendJson = (statusCode, data) => {
      res.writeHead(statusCode, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(data));
    };

    // Helper to read request body
    const readBody = (callback) => {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', () => {
        try {
          const data = body ? JSON.parse(body) : {};
          callback(null, data);
        } catch (err) {
          callback(err);
        }
      });
    };

    // GET /api/cases
    if (pathname === '/api/cases' && method === 'GET') {
      return sendJson(200, {
        cases: [
          { id: 'ieee14', name: 'IEEE 14-Bus Test System', buses: 14, branches: 20, gens: 5 },
          { id: 'ieee30', name: 'IEEE 30-Bus System', buses: 30, branches: 41, gens: 6 }
        ]
      });
    }

    // GET /api/cases/ieee14
    if (pathname === '/api/cases/ieee14' && method === 'GET') {
      try {
        const c = getIEEE14Case();
        return sendJson(200, c);
      } catch (err) {
        return sendJson(500, { error: err.message });
      }
    }

    // GET /api/cases/ieee30
    if (pathname === '/api/cases/ieee30' && method === 'GET') {
      try {
        const c = getIEEE30Case();
        return sendJson(200, c);
      } catch (err) {
        return sendJson(500, { error: err.message });
      }
    }

    // POST /api/cases/synthetic
    if (pathname === '/api/cases/synthetic' && method === 'POST') {
      return readBody((err, data) => {
        if (err) return sendJson(400, { error: 'Invalid JSON body' });
        const n = Math.min(2000, Math.max(10, data.buses || 100));
        const seed = data.seed || 2026;
        const grid = generateSyntheticGrid(n, seed);
        return sendJson(200, grid);
      });
    }

    // POST /api/solve (Remote Power Flow Solve)
    if (pathname === '/api/solve' && method === 'POST') {
      return readBody(async (err, data) => {
        if (err || !data.grid) return sendJson(400, { error: 'Invalid grid data' });
        try {
          const result = await executeRemoteCudaPowerFlow(data.grid, data.options || {});
          return sendJson(200, result);
        } catch (solveErr) {
          return sendJson(500, { error: solveErr.message });
        }
      });
    }

    // POST /api/contingency (Remote N-1 Contingency Screening)
    if (pathname === '/api/contingency' && method === 'POST') {
      return readBody((err, data) => {
        if (err || !data.grid) return sendJson(400, { error: 'Invalid grid data' });
        try {
          const result = runNMinus1Analysis(data.grid, data.options || {});
          return sendJson(200, result);
        } catch (contErr) {
          return sendJson(500, { error: contErr.message });
        }
      });
    }

    // GET /api/gpu/status
    if (pathname === '/api/gpu/status' && method === 'GET') {
      return sendJson(200, getGpuStatus());
    }

    // POST /api/gpu/config
    if (pathname === '/api/gpu/config' && method === 'POST') {
      return readBody(async (err, data) => {
        if (err) return sendJson(400, { error: 'Invalid config body' });
        if (data.profile) setGpuProfile(data.profile);
        if (data.endpoint) await setRemoteEndpoint(data.endpoint);
        return sendJson(200, getGpuStatus());
      });
    }

    // GET /api/benchmarks
    if (pathname === '/api/benchmarks' && method === 'GET') {
      return sendJson(200, loadBenchmarkData());
    }

    return sendJson(404, { error: 'API endpoint not found' });
  }

  // --- STATIC FILE SERVING ---
  let filePath = path.join(PUBLIC_DIR, pathname === '/' ? 'index.html' : pathname);
  const ext = path.extname(filePath).toLowerCase();

  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      filePath = path.join(PUBLIC_DIR, 'index.html');
    }

    const contentType = MIME_TYPES[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
    fs.readFile(filePath, (readErr, content) => {
      if (readErr) {
        res.writeHead(500);
        return res.end('Server file read error');
      }
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    });
  });
});

/**
 * Handle RFC 6455 WebSocket Upgrade & Protocol
 */
server.on('upgrade', (req, socket, head) => {
  const parsedUrl = url.parse(req.url);
  if (parsedUrl.pathname !== '/ws/simulation') {
    socket.destroy();
    return;
  }

  const key = req.headers['sec-websocket-key'];
  if (!key) {
    socket.destroy();
    return;
  }

  // Calculate Sec-WebSocket-Accept hash
  const acceptKey = crypto
    .createHash('sha1')
    .update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11')
    .digest('base64');

  const headers = [
    'HTTP/1.1 101 Switching Protocols',
    'Upgrade: websocket',
    'Connection: Upgrade',
    `Sec-WebSocket-Accept: ${acceptKey}`
  ];

  socket.write(headers.join('\r\n') + '\r\n\r\n');

  // Client connection object
  const client = {
    socket,
    send: (obj) => {
      try {
        const payload = Buffer.from(JSON.stringify(obj), 'utf8');
        const frame = createWebSocketFrame(payload);
        socket.write(frame);
      } catch (err) {
        console.error('WS send error:', err.message);
      }
    }
  };

  wsClients.add(client);

  // Send initial handshake acknowledgement
  client.send({
    type: 'CONNECTED',
    serverTime: Date.now(),
    gpu: getGpuStatus(),
    message: 'Connected to Remote Cloud GPU Power Flow Server'
  });

  // Parse incoming WebSocket frames
  let buffer = Buffer.alloc(0);
  socket.on('data', (chunk) => {
    buffer = Buffer.concat([buffer, chunk]);
    while (buffer.length >= 2) {
      const firstByte = buffer[0];
      const secondByte = buffer[1];
      const isMasked = (secondByte & 0x80) !== 0;
      let payloadLen = secondByte & 0x7F;

      let offset = 2;
      if (payloadLen === 126) {
        if (buffer.length < 4) break;
        payloadLen = buffer.readUInt16BE(2);
        offset = 4;
      } else if (payloadLen === 127) {
        if (buffer.length < 10) break;
        payloadLen = Number(buffer.readBigUInt64BE(2));
        offset = 10;
      }

      const maskLength = isMasked ? 4 : 0;
      if (buffer.length < offset + maskLength + payloadLen) {
        break; // Wait for full frame
      }

      let maskKey = null;
      if (isMasked) {
        maskKey = buffer.slice(offset, offset + 4);
        offset += 4;
      }

      const payload = buffer.slice(offset, offset + payloadLen);
      buffer = buffer.slice(offset + payloadLen);

      // Unmask
      if (isMasked && maskKey) {
        for (let i = 0; i < payload.length; i++) {
          payload[i] ^= maskKey[i % 4];
        }
      }

      const opcode = firstByte & 0x0F;
      if (opcode === 0x8) {
        // Connection close
        socket.end();
        break;
      } else if (opcode === 0x9) {
        // Ping -> respond with Pong (0xA)
        socket.write(Buffer.from([0x8A, 0x00]));
      } else if (opcode === 0x1) {
        // Text frame
        try {
          const msg = JSON.parse(payload.toString('utf8'));
          handleClientWebSocketMessage(client, msg);
        } catch (e) {
          client.send({ type: 'ERROR', error: 'Malformed JSON payload' });
        }
      }
    }
  });

  socket.on('close', () => wsClients.delete(client));
  socket.on('error', () => wsClients.delete(client));
});

/**
 * Creates RFC 6455 unmasked text frame from server to client
 */
function createWebSocketFrame(payload) {
  const len = payload.length;
  let header;
  if (len <= 125) {
    header = Buffer.from([0x81, len]);
  } else if (len <= 65535) {
    header = Buffer.alloc(4);
    header[0] = 0x81;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  return Buffer.concat([header, payload]);
}

/**
 * Handles incoming WebSocket actions
 */
async function handleClientWebSocketMessage(client, msg) {
  if (msg.type === 'PING') {
    client.send({ type: 'PONG', clientTime: msg.clientTime, serverTime: Date.now() });
  } else if (msg.type === 'RUN_SOLVE') {
    client.send({ type: 'SOLVE_STARTED', requestId: msg.requestId });
    try {
      const solution = await executeRemoteCudaPowerFlow(msg.grid, msg.options || {}, (stageEvent) => {
        client.send({
          type: 'PIPELINE_STAGE',
          requestId: msg.requestId,
          event: stageEvent
        });
      });
      client.send({
        type: 'SOLVE_COMPLETE',
        requestId: msg.requestId,
        result: solution
      });
    } catch (err) {
      client.send({
        type: 'SOLVE_FAILED',
        requestId: msg.requestId,
        error: err.message
      });
    }
  } else if (msg.type === 'RUN_CONTINGENCY') {
    client.send({ type: 'CONTINGENCY_STARTED', requestId: msg.requestId });
    try {
      const report = runNMinus1Analysis(msg.grid, msg.options || {});
      client.send({
        type: 'CONTINGENCY_COMPLETE',
        requestId: msg.requestId,
        result: report
      });
    } catch (err) {
      client.send({
        type: 'CONTINGENCY_FAILED',
        requestId: msg.requestId,
        error: err.message
      });
    }
  }
}

/**
 * Open target URL in user's default browser (cross-platform)
 */
function openBrowser(targetUrl) {
  const plat = process.platform;
  let cmd = '';
  if (plat === 'darwin') {
    cmd = `open "${targetUrl}"`;
  } else if (plat === 'win32') {
    cmd = `start "" "${targetUrl}"`;
  } else {
    cmd = `xdg-open "${targetUrl}" > /dev/null 2>&1`;
  }

  const { exec } = require('child_process');
  exec(cmd, () => {});
}

function startServer(portToTry) {
  server.once('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      console.log(`Port ${portToTry} is in use, retrying on http://localhost:${portToTry + 1}...`);
      startServer(portToTry + 1);
    } else {
      console.error('Server listen error:', err.message);
    }
  });

  server.listen(portToTry, '127.0.0.1', () => {
    const url = `http://localhost:${portToTry}`;
    console.log(`================================================================`);
    console.log(`  CUDA POWER FLOW SIMULATION LAB - CLOUD COMPUTE GATEWAY`);
    console.log(`  Web Dashboard:       ${url}`);
    console.log(`  WebSocket Stream:    ws://localhost:${portToTry}/ws/simulation`);
    console.log(`  Architecture:        Browser -> WS/REST -> Cloud Server -> CUDA`);
    console.log(`================================================================`);
    console.log(`  Simulation lab is live. Press Ctrl+C to stop.`);

    if (process.argv.includes('--open') || process.env.AUTO_OPEN === '1') {
      console.log(`  Opening browser: ${url}`);
      openBrowser(url);
    }
  });
}

startServer(PORT);
