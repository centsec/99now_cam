const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const https = require('https');
const crypto = require('crypto');
const { WebSocketServer } = require('ws');
const Anthropic = require('@anthropic-ai/sdk');

// In production (Render sets PORT), HTTPS is terminated at the edge — use plain HTTP.
// Locally, use self-signed HTTPS so the phone can access getUserMedia.
const IS_PRODUCTION = !!process.env.PORT;
const PORT = parseInt(process.env.PORT, 10) || 8443;
const CERT_PATH = path.join(__dirname, '.cert.json');

// --- Claude AI triage ---
const claude = process.env.ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
  : null;
if (!claude) {
  console.warn('[99NOW] ANTHROPIC_API_KEY not set — AI triage will return 503 and dispatch will fall back to keyword mode.');
}

const AI_SYSTEM_PROMPT = `You are the AI clinical-triage assistant for the 99NOW emergency response platform. You provide DECISION SUPPORT to a human 999 call handler; the handler makes every dispatch decision. You do not decide, dispatch, diagnose, or replace clinical judgement.

Given a live transcript of a 999 emergency call in the United Kingdom, produce a structured JSON assessment that helps the handler act faster and more accurately.

CORE PRINCIPLES
1. Advisory only — the handler decides.
2. Reflect real uncertainty in confidence scores. If the transcript is short or ambiguous, say so and keep confidences moderate.
3. Never fabricate. Only reference indicators actually present in the transcript.
4. Use UK 999 / NHS 111 / JRCALC / NHS Pathways terminology and Category 1–4 conventions.
5. Prefer clarity and brevity over verbosity. Every field will be read on a live console under time pressure.

SERVICES YOU CAN RECOMMEND
- police — crimes in progress, violence, weapons, public order, suspected abduction
- fire — fires, gas/chemical, entrapment (RTC entrapment, collapse), rescue
- ambulance — medical emergencies, injuries, cardiac events, trauma
- coastguard — maritime, cliff, tidal, beach, missing at sea

PRIORITY CATEGORIES (aligned with UK ambulance response categories)
- 1 — CATEGORY 1: Immediate threat to life (cardiac arrest, not breathing, active violence with weapon, structure fire with occupants, severe haemorrhage)
- 2 — CATEGORY 2: Serious/emergency (RTC with injuries, chest pain suggestive of ACS, stroke, moderate bleeding)
- 3 — CATEGORY 3: Urgent (falls without severe injury, non-severe illness needing hospital)
- 4 — CATEGORY 4: Less urgent / advice / deferrable

OUTPUT
Return JSON matching the schema. Rules per field:

services[]: exactly the four service keys (police, fire, ambulance, coastguard), each with:
  - confidence: 0–100 integer, calibrated realistically (do not push everything to 90+)
  - reasoning: one sentence, professional tone
  - keyIndicators: exact phrases from the transcript that support the recommendation (may be empty)

priority: the highest category the transcript reasonably supports.
  - category: 1|2|3|4
  - label: e.g. "CATEGORY 1 — IMMEDIATE THREAT TO LIFE"
  - justification: one short sentence

followUpQuestions[]: 3–5 questions the handler could ask NEXT that would materially change response (dispatch category, unit type, pre-arrival instructions). Order by triage value.

preArrivalInstructions[]: 1–4 concrete actions the handler could relay to the caller RIGHT NOW. Safe, non-diagnostic, based on JRCALC/UK first-aid conventions (e.g. "Move casualty into recovery position if breathing but unconscious"; "Begin hands-only CPR at 100–120 bpm if not breathing").

riskFlags[]: real risks the handler should be aware of (child involved, weapon on scene, unresponsive patient, hazmat, potential structural collapse, multiple casualties, hostile environment). Empty array if none. Do not invent.

handoverBrief: 2–3 sentence professional summary suitable to hand to the responding unit. No AI hedging language. Plain, factual, actionable.

Never diagnose specific medical conditions. Never speculate beyond the transcript. If the transcript is empty or single-line, return low confidences and defer with a follow-up question set.`;

const AI_OUTPUT_SCHEMA = {
  type: 'object',
  properties: {
    services: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          key: { type: 'string', enum: ['police', 'fire', 'ambulance', 'coastguard'] },
          name: { type: 'string' },
          confidence: { type: 'integer' },
          reasoning: { type: 'string' },
          keyIndicators: { type: 'array', items: { type: 'string' } },
        },
        required: ['key', 'name', 'confidence', 'reasoning', 'keyIndicators'],
        additionalProperties: false,
      },
    },
    priority: {
      type: 'object',
      properties: {
        category: { type: 'integer', enum: [1, 2, 3, 4] },
        label: { type: 'string' },
        justification: { type: 'string' },
      },
      required: ['category', 'label', 'justification'],
      additionalProperties: false,
    },
    followUpQuestions: { type: 'array', items: { type: 'string' } },
    preArrivalInstructions: { type: 'array', items: { type: 'string' } },
    riskFlags: { type: 'array', items: { type: 'string' } },
    handoverBrief: { type: 'string' },
  },
  required: ['services', 'priority', 'followUpQuestions', 'preArrivalInstructions', 'riskFlags', 'handoverBrief'],
  additionalProperties: false,
};

async function analyseWithClaude({ transcript, geo }) {
  const userMessage =
    `Live 999 call transcript so far:\n\n${transcript || '(no transcript yet)'}\n\n` +
    (geo ? `Caller GPS: lat ${geo.lat.toFixed(5)}, lon ${geo.lon.toFixed(5)} (±${Math.round(geo.accuracy || 0)}m)\n\n` : '') +
    `Produce the structured triage assessment now.`;

  const response = await claude.messages.create({
    model: 'claude-sonnet-5',
    max_tokens: 4096,
    output_config: {
      effort: 'medium',
      format: { type: 'json_schema', schema: AI_OUTPUT_SCHEMA },
    },
    system: [
      { type: 'text', text: AI_SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } },
    ],
    messages: [{ role: 'user', content: userMessage }],
  });

  // With output_config.format the response's first text block is guaranteed valid JSON.
  const textBlock = response.content.find((b) => b.type === 'text');
  if (!textBlock) throw new Error('AI response had no text block');
  return { result: JSON.parse(textBlock.text), usage: response.usage };
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => {
      data += chunk;
      if (data.length > 1_000_000) reject(new Error('Request body too large'));
    });
    req.on('end', () => {
      try { resolve(data ? JSON.parse(data) : {}); } catch (err) { reject(err); }
    });
    req.on('error', reject);
  });
}

function loadOrCreateCert() {
  const selfsigned = require('selfsigned'); // lazy require — not needed in production
  if (fs.existsSync(CERT_PATH)) {
    return JSON.parse(fs.readFileSync(CERT_PATH, 'utf8'));
  }
  const pems = selfsigned.generate(
    [{ name: 'commonName', value: 'phone-cam.local' }],
    { days: 365, keySize: 2048, algorithm: 'sha256' }
  );
  const cert = { key: pems.private, cert: pems.cert };
  fs.writeFileSync(CERT_PATH, JSON.stringify(cert));
  return cert;
}

function getLanAddresses() {
  const out = [];
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const info of ifaces[name]) {
      if (info.family === 'IPv4' && !info.internal) {
        out.push({ name, address: info.address });
      }
    }
  }
  return out;
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

function serveStatic(req, res, file) {
  const full = path.join(__dirname, 'public', file);
  fs.readFile(full, (err, data) => {
    if (err) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(full)] || 'application/octet-stream' });
    res.end(data);
  });
}

const requestHandler = async (req, res) => {
  const url = new URL(req.url, `https://${req.headers.host}`);
  if (url.pathname === '/' || url.pathname === '/index.html') {
    serveStatic(req, res, 'index.html');
    return;
  }
  if (url.pathname === '/phone') { serveStatic(req, res, 'phone.html'); return; }
  if (url.pathname === '/dispatch') { serveStatic(req, res, 'dispatch.html'); return; }
  if (url.pathname === '/responder') { serveStatic(req, res, 'responder.html'); return; }

  if (url.pathname === '/api/analyse' && req.method === 'POST') {
    if (!claude) {
      res.writeHead(503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'ANTHROPIC_API_KEY not configured on server' }));
      return;
    }
    try {
      const payload = await readJsonBody(req);
      const { result, usage } = await analyseWithClaude(payload);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ result, usage, model: 'claude-sonnet-5' }));
    } catch (err) {
      console.error('[99NOW] AI analysis failed:', err.message);
      const status = err instanceof Anthropic.APIError ? (err.status || 500) : 500;
      res.writeHead(status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  res.writeHead(404);
  res.end('not found');
};

let server;
if (IS_PRODUCTION) {
  // Render terminates TLS at their edge; app serves plain HTTP internally.
  server = http.createServer(requestHandler);
} else {
  const cert = loadOrCreateCert();
  server = https.createServer({ key: cert.key, cert: cert.cert }, requestHandler);
}

// --- signaling ---
// Connections: phone (singleton), viewer (singleton), responder-XXXX (many)
// Each ws gets an id. Messages with a `to: <id>` are routed to that peer.
// Messages without `to` are relayed phone<->viewer (legacy behaviour).
const wss = new WebSocketServer({ server });
const peers = new Map(); // id -> ws

function send(ws, msg) {
  if (ws && ws.readyState === ws.OPEN) ws.send(JSON.stringify(msg));
}
function sendTo(id, msg) { send(peers.get(id), msg); }

wss.on('connection', (ws) => {
  ws.id = null; ws.role = null;

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }

    if (msg.type === 'hello') {
      const role = msg.role;
      if (!['viewer', 'phone', 'responder'].includes(role)) return;

      let id;
      if (role === 'responder') {
        id = `responder-${crypto.randomBytes(4).toString('hex')}`;
      } else {
        id = role; // singleton
        if (peers.has(id)) try { peers.get(id).close(); } catch {}
      }
      ws.id = id; ws.role = role;
      peers.set(id, ws);
      console.log(`[${id}] connected (${role})`);

      send(ws, { type: 'welcome', id });

      if (role === 'phone' || role === 'viewer') {
        const other = role === 'phone' ? 'viewer' : 'phone';
        sendTo(other, { type: 'peer-joined' });
        send(ws, { type: 'peer-status', present: peers.has(other) });
      } else if (role === 'responder') {
        // tell the dashboard that a responder connected
        sendTo('viewer', { type: 'responder-joined', id, label: msg.label || id });
      }
      return;
    }

    // Routed message
    if (msg.to) {
      msg.from = ws.id;
      sendTo(msg.to, msg);
      return;
    }

    // Legacy fall-through: phone <-> viewer relay
    if (ws.role === 'phone' || ws.role === 'viewer') {
      const other = ws.role === 'viewer' ? 'phone' : 'viewer';
      sendTo(other, msg);
    }
  });

  ws.on('close', () => {
    if (!ws.id) return;
    peers.delete(ws.id);
    console.log(`[${ws.id}] disconnected`);
    if (ws.role === 'phone') {
      sendTo('viewer', { type: 'peer-left' });
    } else if (ws.role === 'viewer') {
      sendTo('phone', { type: 'peer-left' });
      for (const [id, p] of peers) {
        if (p.role === 'responder') sendTo(id, { type: 'viewer-left' });
      }
    } else if (ws.role === 'responder') {
      sendTo('viewer', { type: 'responder-left', id: ws.id });
    }
  });
});

server.listen(PORT, '0.0.0.0', () => {
  if (IS_PRODUCTION) {
    console.log(`99NOW server running on port ${PORT} (production mode)`);
    return;
  }
  const addrs = getLanAddresses();
  console.log(`\n99NOW server running on port ${PORT}`);
  console.log('\nOpen the dashboard on your laptop:');
  for (const { name, address } of addrs) {
    console.log(`  https://${address}:${PORT}/   (${name})`);
  }
  console.log('  https://localhost:' + PORT + '/');
  console.log('\nClick through the self-signed cert warning on both laptop and phone.\n');
});
