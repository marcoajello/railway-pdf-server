// Version 3.0.0 - AI-powered border fix detection (context-aware)

const express = require('express');
const puppeteer = require('puppeteer');
const cors = require('cors');
const multer = require('multer');
const Anthropic = require('@anthropic-ai/sdk');
const sharp = require('sharp');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

// Face-API.js setup
let faceapi = null;
let faceApiReady = false;

async function initFaceApi() {
  try {
    // Dynamic import for face-api
    const faceapiModule = await import('@vladmandic/face-api');
    faceapi = faceapiModule.default || faceapiModule;
    
    // Setup canvas environment for Node.js
    const canvas = await import('canvas');
    const { Canvas, Image, ImageData } = canvas.default || canvas;
    faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
    
    // Load models from local directory
    const modelsPath = path.join(__dirname, 'models');
    
    // Check if models exist, if not use CDN path
    let modelUri = modelsPath;
    try {
      await fs.access(modelsPath);
      console.log('[FaceAPI] Loading models from local path:', modelsPath);
    } catch {
      // Models not found locally - they need to be downloaded
      console.log('[FaceAPI] Models directory not found at:', modelsPath);
      console.log('[FaceAPI] Please download models from https://github.com/vladmandic/face-api/tree/master/model');
      return false;
    }
    
    await faceapi.nets.tinyFaceDetector.loadFromDisk(modelUri);
    await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(modelUri);
    
    faceApiReady = true;
    console.log('[FaceAPI] Models loaded successfully');
    return true;
  } catch (err) {
    console.error('[FaceAPI] Init error:', err.message);
    faceApiReady = false;
    return false;
  }
}

// Initialize face-api on startup
initFaceApi();

const app = express();
const PORT = process.env.PORT || 3000;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ['application/pdf', 'image/jpeg', 'image/png', 'image/webp'];
    cb(null, allowed.includes(file.mimetype));
  }
});

let anthropic = null;
function getAnthropicClient() {
  if (!anthropic && process.env.ANTHROPIC_API_KEY) {
    anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  }
  return anthropic;
}

app.use(cors());
app.use(express.json({ limit: '150mb' }));

// Auth callback endpoint - handles email confirmation redirect
// Supabase redirects here with tokens, we pass them to the Electron app
app.get('/auth/callback', (req, res) => {
  // Supabase sends tokens as hash fragments, but for email confirmation
  // it sends them as query params when using PKCE or as hash
  // We'll serve a page that extracts the hash and redirects to the app
  
  res.send(`
<!DOCTYPE html>
<html>
<head>
  <title>Signing you in...</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #1a1a1a;
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
    }
    .container { text-align: center; }
    .spinner {
      width: 40px;
      height: 40px;
      border: 3px solid #333;
      border-top-color: #4a90d9;
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin: 0 auto 20px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    a { color: #4a90d9; }
  </style>
</head>
<body>
  <div class="container">
    <div class="spinner"></div>
    <p id="status">Opening Skeduler...</p>
    <p id="fallback" style="display:none; margin-top: 20px;">
      <a href="#" id="manualLink">Click here to open Skeduler</a><br><br>
      <span style="color: #666; font-size: 13px;">If the app doesn't open, launch it manually and sign in.</span>
    </p>
  </div>
  <script>
    // Get tokens from URL hash (Supabase sends them as hash fragments)
    const hash = window.location.hash.substring(1);
    const params = new URLSearchParams(hash);
    
    const accessToken = params.get('access_token');
    const refreshToken = params.get('refresh_token');
    const error = params.get('error');
    const errorDesc = params.get('error_description');
    
    if (error) {
      document.getElementById('status').textContent = errorDesc || error;
      document.getElementById('status').style.color = '#ff6b6b';
    } else if (accessToken) {
      // Build deep link with tokens
      const deepLink = 'skeduler://auth?access_token=' + encodeURIComponent(accessToken) + 
                       '&refresh_token=' + encodeURIComponent(refreshToken || '');
      
      // Try to open the app
      window.location.href = deepLink;
      
      // Show fallback after 2 seconds
      setTimeout(() => {
        document.getElementById('status').textContent = 'Waiting for app to open...';
        document.getElementById('fallback').style.display = 'block';
        document.getElementById('manualLink').href = deepLink;
      }, 2000);
    } else {
      document.getElementById('status').textContent = 'No authentication token found.';
      document.getElementById('status').style.color = '#ff6b6b';
    }
  </script>
</body>
</html>
  `);
});

app.get('/', (req, res) => {
  res.json({ status: 'ok', version: '3.0.0', features: ['pdf-generation', 'html-generation', 'storyboard-extraction', 'ai-border-fix'] });
});

// PDF generation endpoint
app.post('/generate-pdf', async (req, res) => {
  let browser = null;
  try {
    const { html, orientation = 'landscape' } = req.body;
    if (!html) return res.status(400).json({ error: 'HTML required' });
    
    browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    });
    
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: ['load', 'networkidle0'] });
    
    const pdfBuffer = await page.pdf({
      format: 'Letter',
      landscape: orientation === 'landscape',
      printBackground: true,
      margin: { top: '0.4in', right: '0.4in', bottom: '0.4in', left: '0.4in' },
      preferCSSPageSize: true
    });
    
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename="schedule.pdf"');
    res.send(pdfBuffer);
  } catch (error) {
    console.error('PDF error:', error);
    res.status(500).json({ error: error.message });
  } finally {
    if (browser) await browser.close();
  }
});

// HTML generation endpoint (for broadcast)
app.post('/generate-html', async (req, res) => {
  let browser = null;
  try {
    const { html } = req.body;
    if (!html) return res.status(400).json({ error: 'HTML required' });
    
    browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    });
    
    const page = await browser.newPage();
    await page.setViewport({ width: 1400, height: 900 });
    await page.setContent(html, { waitUntil: ['load', 'networkidle0'] });
    
    const renderedHtml = await page.evaluate(() => {
      document.querySelectorAll('script').forEach(s => s.remove());
      document.querySelectorAll('button, input[type="checkbox"]').forEach(el => el.remove());
      document.querySelectorAll('[contenteditable]').forEach(el => el.removeAttribute('contenteditable'));
      document.querySelectorAll('[draggable]').forEach(el => el.removeAttribute('draggable'));
      return document.documentElement.outerHTML;
    });
    
    const fullHtml = `<!DOCTYPE html>\n${renderedHtml}`;
    
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.send(fullHtml);
    
  } catch (error) {
    console.error('HTML generation error:', error);
    res.status(500).json({ error: error.message });
  } finally {
    if (browser) await browser.close();
  }
});

// ============================================
// HANGING CHAD DETECTION (AI-powered)
// ============================================

// ============================================================
// PDF Page Analysis Cache
// ============================================================
// Caches analysis results to avoid redundant API calls.
// Key: hash of schedule content + settings
// Value: { pages, totalRows, timestamp }
// TTL: 24 hours (schedule changes invalidate the hash anyway)

const analysisCache = new Map();
const CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

function cleanExpiredCache() {
  const now = Date.now();
  for (const [key, value] of analysisCache.entries()) {
    if (now - value.timestamp > CACHE_TTL_MS) {
      analysisCache.delete(key);
    }
  }
}

// Clean cache every hour
setInterval(cleanExpiredCache, 60 * 60 * 1000);

/**
 * Analyze PDF page bottoms for hanging borders using Claude Vision
 * POST /api/analyze-pdf-pages
 * Body: { images: [...], contentHash: "optional-hash-for-caching" }
 */
app.post('/api/analyze-pdf-pages', async (req, res) => {
  try {
    const { images, contentHash } = req.body;
    
    // Check cache first if contentHash provided
    if (contentHash) {
      const cached = analysisCache.get(contentHash);
      if (cached && (Date.now() - cached.timestamp < CACHE_TTL_MS)) {
        console.log(`[PDF Analyze] Cache HIT for hash: ${contentHash.substring(0, 16)}...`);
        return res.json({
          success: true,
          pages: cached.pages,
          totalRows: cached.totalRows,
          hasIssues: cached.pages.length > 0,
          cached: true
        });
      }
      console.log(`[PDF Analyze] Cache MISS for hash: ${contentHash.substring(0, 16)}...`);
    }
    
    if (!images || !Array.isArray(images)) {
      return res.status(400).json({ error: 'Missing or invalid images array' });
    }
    
    const client = getAnthropicClient();
    if (!client) {
      return res.status(500).json({ error: 'ANTHROPIC_API_KEY not set' });
    }
    
    console.log(`[PDF Analyze] Analyzing ${images.length} page bottoms...`);
    
    // Build image content array for Claude
    const imageContents = images.map(img => [
      {
        type: 'text',
        text: `Page ${img.pageNum}:`
      },
      {
        type: 'image',
        source: {
          type: 'base64',
          media_type: 'image/png',
          // Handle both raw base64 and data URL format
          data: img.dataUrl.includes(',') ? img.dataUrl.split(',')[1] : img.dataUrl
        }
      }
    ]).flat();
    
    const response = await client.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1000,
      messages: [{
        role: 'user',
        content: [
          {
            type: 'text',
            text: `You are analyzing PDF pages of a schedule table. Count the DATA rows on each page.

DATA ROWS are the white rows with content (times, descriptions, images). 
DO NOT count the gray HEADER row (START-END, DUR, #, DESCRIPTION, etc).
Row 1 = the first data row after the header.

For EACH page image, count how many data rows are visible (even partially).

RESPOND WITH ONLY JSON:
{
  "pages": [
    {"page": 1, "rowCount": 7},
    {"page": 2, "rowCount": 4}
  ]
}

This tells me: Page 1 has rows 1-7, Page 2 has rows 8-11.`
          },
          ...imageContents
        ]
      }]
    });
    
    const text = response.content[0]?.text || '{"pages":[]}';
    
    console.log('[PDF Analyze] Raw Claude response:', text.substring(0, 500));
    
    // Parse JSON response
    let cleanText = text.replace(/```json\s*/g, '').replace(/```\s*/g, '');
    
    // Try to find JSON object in the response
    const jsonMatch = cleanText.match(/\{[\s\S]*"pages"[\s\S]*\}/);
    if (jsonMatch) {
      cleanText = jsonMatch[0];
    }
    
    let result;
    try {
      result = JSON.parse(cleanText.trim());
    } catch (parseError) {
      console.error('[PDF Analyze] JSON parse failed, raw text:', text);
      return res.json({
        success: true,
        pages: [],
        hasIssues: false
      });
    }
    
    const pages = result.pages || [];
    
    // Calculate row ranges from counts
    // e.g., [{page:1, rowCount:7}, {page:2, rowCount:4}] 
    // becomes [{page:1, startRow:1, endRow:7}, {page:2, startRow:8, endRow:11}]
    let currentRow = 1;
    const pageRanges = pages.map(p => {
      const range = {
        page: p.page,
        startRow: currentRow,
        endRow: currentRow + p.rowCount - 1,
        rowCount: p.rowCount
      };
      currentRow += p.rowCount;
      return range;
    });
    
    console.log(`[PDF Analyze] Page ranges:`, pageRanges);
    
    // Cache the result if contentHash was provided
    if (contentHash) {
      analysisCache.set(contentHash, {
        pages: pageRanges,
        totalRows: currentRow - 1,
        timestamp: Date.now()
      });
      console.log(`[PDF Analyze] Cached result for hash: ${contentHash.substring(0, 16)}... (cache size: ${analysisCache.size})`);
    }
    
    return res.json({
      success: true,
      pages: pageRanges,
      totalRows: currentRow - 1,
      hasIssues: pages.length > 0
    });
    
  } catch (error) {
    console.error('[PDF Analyze] Error:', error);
    return res.status(500).json({ 
      error: 'Analysis failed',
      message: error.message 
    });
  }
});

// ============================================
// STORYBOARD EXTRACTION
// ============================================

/**
 * Detect rectangles using Python/OpenCV
 */
async function detectRectangles(imagePath) {
  return new Promise((resolve) => {
    const script = path.join(__dirname, 'frame_detector.py');
    
    const tryPython = (cmd) => {
      const proc = spawn(cmd, [script, imagePath, 'crop']);
      
      let stdout = '';
      let stderr = '';
      
      proc.stdout.on('data', d => stdout += d);
      proc.stderr.on('data', d => stderr += d);
      
      proc.on('close', code => {
        if (code !== 0) {
          console.error(`[Storyboard] ${cmd} error (code ${code}):`, stderr);
          if (cmd === 'python3') {
            console.log('[Storyboard] Trying python instead...');
            tryPython('python');
            return;
          }
          resolve({ count: 0, rectangles: [], images: [] });
          return;
        }
        try {
          const result = JSON.parse(stdout);
          console.log(`[Storyboard] Found ${result.count} rectangles`);
          resolve(result);
        } catch (e) {
          console.error('[Storyboard] JSON parse error');
          resolve({ count: 0, rectangles: [], images: [] });
        }
      });
      
      proc.on('error', err => {
        console.error(`[Storyboard] ${cmd} spawn error:`, err.message);
        if (cmd === 'python3') {
          console.log('[Storyboard] Trying python instead...');
          tryPython('python');
          return;
        }
        resolve({ count: 0, rectangles: [], images: [] });
      });
    };
    
    tryPython('python3');
  });
}

/**
 * Convert PDF to page images with retry logic for CDN reliability
 */
async function pdfToImages(pdfBuffer, outputDir, retryCount = 0) {
  const MAX_RETRIES = 2;
  let browser = null;
  const images = [];
  
  try {
    browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    });
    
    const page = await browser.newPage();
    page.setDefaultNavigationTimeout(180000);
    page.setDefaultTimeout(180000);
    await page.setViewport({ width: 1200, height: 1600, deviceScaleFactor: 2 });
    
    const base64Pdf = pdfBuffer.toString('base64');
    
    // Inline PDF.js to avoid CDN dependency issues
    // We fetch it once and embed it directly
    const html = `<!DOCTYPE html><html><head>
      <style>body{margin:0;background:white}canvas{display:block}#status{position:fixed;top:10px;left:10px;font-family:sans-serif;}</style>
    </head><body>
      <div id="status">Loading PDF.js...</div>
      <canvas id="canvas"></canvas>
      <script>
        window.pdfError = null;
        window.pdfReady = false;
        
        // Load PDF.js with error handling
        function loadScript(url, callback) {
          var script = document.createElement('script');
          script.type = 'text/javascript';
          script.onload = callback;
          script.onerror = function() {
            window.pdfError = 'Failed to load: ' + url;
            document.getElementById('status').textContent = window.pdfError;
          };
          script.src = url;
          document.head.appendChild(script);
        }
        
        loadScript('https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js', function() {
          document.getElementById('status').textContent = 'PDF.js loaded, configuring...';
          pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
          window.pdfReady = true;
          document.getElementById('status').textContent = 'Ready';
        });
        
        window.renderPage = async function(n) {
          const d = atob('${base64Pdf}');
          const a = new Uint8Array(d.length);
          for (let i = 0; i < d.length; i++) a[i] = d.charCodeAt(i);
          const pdf = await pdfjsLib.getDocument({data: a}).promise;
          if (n > pdf.numPages) return null;
          const pg = await pdf.getPage(n);
          const vp = pg.getViewport({scale: 2});
          const c = document.getElementById('canvas');
          c.width = vp.width;
          c.height = vp.height;
          await pg.render({canvasContext: c.getContext('2d'), viewport: vp}).promise;
          return {w: c.width, h: c.height};
        };
        
        window.getPageCount = async function() {
          const d = atob('${base64Pdf}');
          const a = new Uint8Array(d.length);
          for (let i = 0; i < d.length; i++) a[i] = d.charCodeAt(i);
          return (await pdfjsLib.getDocument({data: a}).promise).numPages;
        };
      </script>
    </body></html>`;
    
    // Use 'load' instead of 'networkidle0' to be more tolerant
    await page.setContent(html, { waitUntil: 'load', timeout: 60000 });
    
    // Wait for PDF.js with polling (more reliable than waitForFunction with CDN)
    let waitAttempts = 0;
    const maxWaitAttempts = 60; // 60 seconds max
    while (waitAttempts < maxWaitAttempts) {
      const status = await page.evaluate(() => ({ ready: window.pdfReady, error: window.pdfError }));
      if (status.error) {
        throw new Error(status.error);
      }
      if (status.ready) {
        break;
      }
      await new Promise(r => setTimeout(r, 1000));
      waitAttempts++;
    }
    
    if (waitAttempts >= maxWaitAttempts) {
      throw new Error('PDF.js load timeout after 60 seconds');
    }
    
    console.log(`[Storyboard] PDF.js loaded after ${waitAttempts}s`);
    
    const pageCount = await page.evaluate('getPageCount()');
    console.log(`[Storyboard] PDF: ${pageCount} pages`);
    
    for (let i = 1; i <= pageCount; i++) {
      await page.evaluate(`renderPage(${i})`);
      await new Promise(r => setTimeout(r, 400));
      const canvas = await page.$('#canvas');
      const tempPath = path.join(outputDir, `page-${i}-temp.png`);
      const imgPath = path.join(outputDir, `page-${i}.jpg`);
      
      await canvas.screenshot({ path: tempPath, type: 'png' });
      
      await sharp(tempPath)
        .resize(1200, 1200, { fit: 'inside', withoutEnlargement: false })
        .jpeg({ quality: 40, progressive: true })
        .toFile(imgPath);
      
      await fs.unlink(tempPath);
      
      images.push(imgPath);
    }
    
    return images;
  } catch (error) {
    console.error(`[Storyboard] pdfToImages error (attempt ${retryCount + 1}):`, error.message);
    
    if (browser) {
      await browser.close();
      browser = null;
    }
    
    // Retry on CDN/timeout errors
    if (retryCount < MAX_RETRIES && (
      error.message.includes('timeout') || 
      error.message.includes('Timeout') ||
      error.message.includes('Failed to load') ||
      error.message.includes('net::')
    )) {
      console.log(`[Storyboard] Retrying... (${retryCount + 1}/${MAX_RETRIES})`);
      await new Promise(r => setTimeout(r, 2000 * (retryCount + 1))); // Exponential backoff
      return pdfToImages(pdfBuffer, outputDir, retryCount + 1);
    }
    
    throw error;
  } finally {
    if (browser) await browser.close();
  }
}

/**
 * Extract text from page using Claude
 */
async function extractText(imageBuffer) {
  const client = getAnthropicClient();
  if (!client) throw new Error('ANTHROPIC_API_KEY not set');
  
  const resized = await sharp(imageBuffer)
    .resize(1400, 1400, { fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 85 })
    .toBuffer();
  
  const response = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 4096,
    messages: [{
      role: 'user',
      content: [
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: resized.toString('base64') } },
        { type: 'text', text: `Extract storyboard data from this page.

STEP 1 - SPOT/SCRIPT NAME (CRITICAL):
Look for a BOLD TITLE near the top of the page - this is the commercial/spot name.
These titles indicate DIFFERENT COMMERCIALS/SPOTS in the same PDF.
Always extract the title exactly as written - even if it's just a number.
This is NOT scene descriptions like "INT. KITCHEN" - those are scene headers within a commercial.

STEP 2 - GRID LAYOUT:
Identify the grid structure. Read frames LEFT-TO-RIGHT, then TOP-TO-BOTTOM.

STEP 3 - FRAME NUMBERS:
- If frames have visible numbers (1, 2, 1A, 1B, etc.), use those exactly
- If NO visible numbers, number sequentially: 1, 2, 3, 4...

STEP 4 - EXTRACT:
Return JSON:
{
  "spotName": "EXACT TITLE FROM PAGE" or null,
  "gridLayout": "2x3",
  "hasVisibleNumbers": true/false,
  "frames": [
    {
      "frameNumber": "1",
      "description": "Action/direction text",
      "dialog": "CHARACTER: Spoken lines..."
    }
  ]
}

RULES:
- spotName: ALWAYS extract the bold title at top of page - this identifies which commercial/spot
- description: action/camera direction text
- dialog: spoken lines with character prefix
- Skip empty frames` }
      ]
    }]
  });
  
  const text = response.content[0].text;
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  try {
    return JSON.parse((jsonMatch ? jsonMatch[1] : text).trim());
  } catch (e) {
    console.error('[Storyboard] Text parse error');
    return { spotName: null, frames: [] };
  }
}

function groupIntoShots(frames) {
  const shots = [];
  let currentShot = null;
  let shotNumber = 1;
  
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    
    // Start new shot if:
    // - First frame
    // - Frame doesn't continue previous
    // - Has visible numbers and number changed (different base number)
    let startNewShot = false;
    
    if (i === 0) {
      startNewShot = true;
    } else if (f.hasVisibleNumber) {
      // If has visible numbers, group by base number (existing logic)
      const prevNum = (frames[i-1].frameNumber || '').replace(/^(FR|FRAME|SHOT)[\s.]*/i, '').match(/^(\d+)/)?.[1];
      const currNum = (f.frameNumber || '').replace(/^(FR|FRAME|SHOT)[\s.]*/i, '').match(/^(\d+)/)?.[1];
      startNewShot = prevNum !== currNum;
    } else {
      // No visible numbers - use AI's grouping (shotGroup field from Pass 2)
      if (f.shotGroup !== undefined && frames[i-1].shotGroup !== undefined) {
        startNewShot = f.shotGroup !== frames[i-1].shotGroup;
      } else {
        // Fallback: each frame is its own shot
        startNewShot = true;
      }
    }
    
    if (startNewShot) {
      if (currentShot) shots.push(currentShot);
      currentShot = {
        shotNumber: String(shotNumber++),
        frames: [],
        images: [],
        descriptions: [],
        dialogs: []
      };
    }
    
    currentShot.frames.push(f.frameNumber);
    if (f.image) currentShot.images.push(f.image);
    if (f.description) currentShot.descriptions.push(f.description);
    if (f.dialog) currentShot.dialogs.push(f.dialog);
  }
  
  // Don't forget the last shot
  if (currentShot) shots.push(currentShot);
  
  return shots.map(g => ({
    shotNumber: g.shotNumber,
    frames: g.frames,
    images: g.images,
    descriptions: g.descriptions,  // Keep per-frame arrays for drag/drop
    dialogs: g.dialogs,            // Keep per-frame arrays for drag/drop
    description: g.descriptions.join('\n'),
    dialog: g.dialogs.join('\n'),
    combined: [...g.descriptions, '', ...g.dialogs].filter(Boolean).join('\n')
  }));
}

// Pass 2: AI-powered shot grouping analysis
async function analyzeGroupings(frames) {
  const client = getAnthropicClient();
  if (!client) return frames; // Fallback to pass 1 results
  
  // Only analyze if we have images and no visible numbers
  const framesWithImages = frames.filter(f => f.image);
  if (framesWithImages.length < 2) return frames;
  if (frames.every(f => f.hasVisibleNumber)) return frames; // Trust visible numbers
  
  try {
    // Build a grid of thumbnails for the AI to analyze
    const imageContents = [];
    
    // Create smaller thumbnails for pass 2
    for (let i = 0; i < Math.min(framesWithImages.length, 24); i++) {
      const f = framesWithImages[i];
      if (f.image) {
        const thumb = await sharp(Buffer.from(f.image, 'base64'))
          .resize(300, 300, { fit: 'inside', withoutEnlargement: true })
          .jpeg({ quality: 70 })
          .toBuffer();
        
        imageContents.push({
          type: 'image',
          source: { type: 'base64', media_type: 'image/jpeg', data: thumb.toString('base64') }
        });
      }
    }
    
    if (imageContents.length < 2) return frames;
    
    console.log(`[Storyboard] Pass 2: Analyzing ${imageContents.length} frames for shot grouping`);
    
    const response = await client.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1024,
      messages: [{
        role: 'user',
        content: [
          ...imageContents,
          { type: 'text', text: `These are ${imageContents.length} storyboard frames in sequence (Frame 1 through Frame ${imageContents.length}).

Group them into SHOTS. Frames belong to the SAME SHOT if:

1. SAME BACKGROUND ARCHITECTURE - Same room, same distinctive elements (range hood, cabinets, windows) visible from same angle
2. SAME CHARACTER ARRANGEMENT - Characters in similar positions relative to each other  
3. SAME CAMERA ANGLE - Shooting from same general direction
4. CONTINUOUS ACTION - Action flows naturally across frames

IMPORTANT: Storyboard artists are inconsistent with scale. Focus on BACKGROUND ELEMENTS and ENVIRONMENT - if the same room/architecture is visible from the same angle, it's likely one continuous shot even if character positions shift slightly.

Frames are DIFFERENT SHOTS if:
- Completely different location or background
- Camera clearly on opposite side of characters
- Cut to different subject entirely (insert, new scene)

BIAS TOWARD GROUPING consecutive frames in the same environment.

Return ONLY a JSON array:
[[1, 2, 3, 4], [5], [6, 7], [8, 9, 10]]` }
        ]
      }]
    });
    
    const text = response.content[0].text;
    const jsonMatch = text.match(/\[[\s\S]*\]/);
    
    if (jsonMatch) {
      const groups = JSON.parse(jsonMatch[0]);
      console.log(`[Storyboard] Pass 2: AI grouped into ${groups.length} shots`);
      
      // Apply groupings to frames
      groups.forEach((group, shotIdx) => {
        group.forEach(frameNum => {
          const idx = frameNum - 1; // Convert to 0-indexed
          if (frames[idx]) {
            frames[idx].shotGroup = shotIdx;
          }
        });
      });
    }
  } catch (e) {
    console.error('[Storyboard] Pass 2 error:', e.message);
    // Fallback to pass 1 results
  }
  
  return frames;
}

app.post('/api/extract-storyboard', upload.single('pdf'), async (req, res) => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'storyboard-'));
  
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    if (!process.env.ANTHROPIC_API_KEY) return res.status(500).json({ error: 'ANTHROPIC_API_KEY not set' });
    
    console.log('[Storyboard] Processing:', req.file.originalname);
    const startTime = Date.now();
    
    const imageDir = path.join(tempDir, 'images');
    await fs.mkdir(imageDir, { recursive: true });
    
    let pageImages = [];
    if (req.file.mimetype === 'application/pdf') {
      pageImages = await pdfToImages(req.file.buffer, imageDir);
    } else {
      const imgPath = path.join(imageDir, 'page-1.png');
      await sharp(req.file.buffer).png().toFile(imgPath);
      pageImages = [imgPath];
    }
    
    console.log(`[Storyboard] ${pageImages.length} page(s) - starting processing`);
    
    // BATCH APPROACH: Process up to 4 pages per API call for efficiency
    const BATCH_SIZE = 4;
    const batches = [];
    for (let i = 0; i < pageImages.length; i += BATCH_SIZE) {
      batches.push(pageImages.slice(i, i + BATCH_SIZE).map((p, j) => ({ path: p, pageNum: i + j + 1 })));
    }
    
    // Process batches with controlled concurrency (2 concurrent batches max)
    const CONCURRENCY = 2;
    const allPageResults = [];
    
    for (let i = 0; i < batches.length; i += CONCURRENCY) {
      const batchGroup = batches.slice(i, i + CONCURRENCY);
      
      const batchResults = await Promise.all(batchGroup.map(async (batch) => {
        // Run rectangle detection in parallel for this batch
        const rectPromises = batch.map(({ path, pageNum }) => 
          detectRectangles(path).then(r => ({ pageNum, detected: r }))
        );
        
        // Run batched text extraction (one API call for multiple pages)
        const textPromise = extractTextBatched(batch);
        
        const [rectResults, textResults] = await Promise.all([
          Promise.all(rectPromises),
          textPromise
        ]);
        
        // Merge results by page
        return batch.map(({ pageNum }) => {
          const detected = rectResults.find(r => r.pageNum === pageNum)?.detected || { count: 0, images: [] };
          const textData = textResults.find(r => r.pageNum === pageNum) || { frames: [] };
          return { detected, textData, pageNum };
        });
      }));
      
      allPageResults.push(...batchResults.flat());
    }
    
    // Sort by page number to maintain order
    allPageResults.sort((a, b) => a.pageNum - b.pageNum);
    
    const allFrames = [];
    let currentSpot = null;
    
    for (const { detected, textData, pageNum } of allPageResults) {
      if (textData.spotName) currentSpot = textData.spotName;
      
      const textFrames = textData.frames || [];
      const images = detected.images || [];
      const hasVisibleNumbers = textData.hasVisibleNumbers === true;
      
      const maxLen = Math.max(textFrames.length, images.length);
      for (let j = 0; j < maxLen; j++) {
        const tf = textFrames[j] || {};
        const img = images[j] || null;
        
        allFrames.push({
          frameNumber: tf.frameNumber || `${j + 1}`,
          hasVisibleNumber: hasVisibleNumbers,
          description: tf.description || '',
          dialog: tf.dialog || '',
          image: img,
          spotName: currentSpot,
          pageNum: pageNum
        });
      }
    }
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[Storyboard] Total: ${allFrames.length} frames, ${allFrames.filter(f => f.image).length} with images (${elapsed}s)`);
    
    // Group frames by spot
    const spotGroups = {};
    for (const f of allFrames) {
      const spot = f.spotName || 'Untitled';
      if (!spotGroups[spot]) spotGroups[spot] = [];
      spotGroups[spot].push(f);
    }
    
    // For each spot, check if we need to renumber, then run pass 2 analysis
    const spots = [];
    for (const [name, frames] of Object.entries(spotGroups)) {
      // Check if any frame lacks visible numbers
      const anyWithoutNumbers = frames.some(f => !f.hasVisibleNumber);
      
      // Check for duplicate frame numbers (same number on different pages = needs renumbering)
      const numberPageMap = {};
      let hasDuplicates = false;
      for (const f of frames) {
        const key = f.frameNumber;
        if (numberPageMap[key] && numberPageMap[key] !== f.pageNum) {
          hasDuplicates = true;
          break;
        }
        numberPageMap[key] = f.pageNum;
      }
      
      // Renumber if needed
      if (anyWithoutNumbers || hasDuplicates) {
        console.log(`[Storyboard] Spot "${name}": renumbering ${frames.length} frames (duplicates: ${hasDuplicates}, missing numbers: ${anyWithoutNumbers})`);
        frames.forEach((f, idx) => {
          f.frameNumber = String(idx + 1);
        });
        
        // Pass 2: AI grouping analysis (only for unnumbered storyboards)
        await analyzeGroupings(frames);
      }
      
      spots.push({
        name,
        shots: groupIntoShots(frames)
      });
    }
    
    res.json({ spots });
    
  } catch (error) {
    console.error('[Storyboard] Error:', error);
    res.status(500).json({ error: error.message });
  } finally {
    // Clean up temp directory
    try { await fs.rm(tempDir, { recursive: true, force: true }); } catch (e) {}
  }
});

/**
 * Extract text from multiple pages in a single API call
 */
async function extractTextBatched(pages) {
  const client = getAnthropicClient();
  if (!client) throw new Error('ANTHROPIC_API_KEY not set');
  
  // Prepare all images
  const imageContents = await Promise.all(pages.map(async ({ path: imagePath, pageNum }) => {
    const imageBuffer = await fs.readFile(imagePath);
    const resized = await sharp(imageBuffer)
      .resize(1200, 1200, { fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 80 })
      .toBuffer();
    
    return {
      pageNum,
      content: {
        type: 'image',
        source: { type: 'base64', media_type: 'image/jpeg', data: resized.toString('base64') }
      }
    };
  }));
  
  // Build message with all images
  const content = [];
  imageContents.forEach(({ pageNum, content: imgContent }) => {
    content.push({ type: 'text', text: `--- PAGE ${pageNum} ---` });
    content.push(imgContent);
  });
  
  content.push({ type: 'text', text: `Extract storyboard data from each page above.

STEP 1 - SPOT/SCRIPT NAME (CRITICAL):
Look for a BOLD TITLE near the top of each page - this is the commercial/spot name.
These titles indicate DIFFERENT COMMERCIALS/SPOTS in the same PDF.
Always extract the title exactly as written - even if it's just a number.
This is NOT scene descriptions like "INT. KITCHEN" - those are scene headers within a commercial.

STEP 2 - GRID LAYOUT:
Identify the grid structure. Read frames LEFT-TO-RIGHT, then TOP-TO-BOTTOM.

STEP 3 - FRAME NUMBERS:
- If frames have visible numbers (1, 2, 1A, 1B, etc.), use those exactly
- If NO visible numbers, number sequentially: 1, 2, 3, 4...

STEP 4 - EXTRACT:
Return a JSON array with one object per page:
[
  {
    "pageNum": 1,
    "spotName": "EXACT TITLE FROM PAGE" or null,
    "gridLayout": "2x3",
    "hasVisibleNumbers": true/false,
    "frames": [
      { "frameNumber": "1", "description": "Action/direction text", "dialog": "CHARACTER: Spoken lines..." }
    ]
  },
  { "pageNum": 2, ... }
]

RULES:
- spotName: ALWAYS extract the bold title at top of page - this identifies which commercial/spot
- description: action/camera direction text
- dialog: spoken lines with character prefix
- Skip empty frames` });
  
  console.log(`[Storyboard] Batched API call for ${pages.length} pages`);
  
  const response = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 8192,
    messages: [{ role: 'user', content }]
  });
  
  const text = response.content[0].text;
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) || text.match(/\[[\s\S]*\]/);
  
  try {
    const parsed = JSON.parse(jsonMatch ? (jsonMatch[1] || jsonMatch[0]).trim() : text.trim());
    // Ensure it's an array
    const results = Array.isArray(parsed) ? parsed : [parsed];
    console.log(`[Storyboard] Batched extraction returned ${results.length} page results`);
    return results;
  } catch (e) {
    console.error('[Storyboard] Batched text parse error:', e.message);
    // Return empty results for each page
    return pages.map(({ pageNum }) => ({ pageNum, spotName: null, frames: [] }));
  }
}

// CAST EXTRACTION
app.post('/api/extract-cast', upload.single('pdf'), async (req, res) => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'cast-'));
  
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    if (!process.env.ANTHROPIC_API_KEY) return res.status(500).json({ error: 'ANTHROPIC_API_KEY not set' });
    
    console.log('[Cast] Processing:', req.file.originalname);
    const startTime = Date.now();
    
    const imageDir = path.join(tempDir, 'images');
    await fs.mkdir(imageDir, { recursive: true });
    
    let pageImages = [];
    if (req.file.mimetype === 'application/pdf') {
      pageImages = await pdfToImages(req.file.buffer, imageDir);
    } else {
      const imgPath = path.join(imageDir, 'page-1.png');
      await sharp(req.file.buffer).png().toFile(imgPath);
      pageImages = [imgPath];
    }
    
    console.log(`[Cast] ${pageImages.length} page(s) - processing`);
    
    const allCast = [];
    
    for (let i = 0; i < pageImages.length; i++) {
      const imagePath = pageImages[i];
      const pageNum = i + 1;
      
      console.log(`[Cast] Page ${pageNum}: analyzing...`);
      
      const imageBuffer = await fs.readFile(imagePath);
      
      // Detect rectangles (headshots)
      const detected = await detectRectangles(imagePath);
      
      // Use AI to extract actor/character names
      const castData = await extractCastText(imageBuffer);
      
      console.log(`[Cast] Page ${pageNum}: ${detected.count} images, ${castData.members?.length || 0} members`);
      
      // Match images to cast members
      const members = castData.members || [];
      const rawImages = detected.images || [];
      
      // Pre-filter: only keep images that contain faces (not text boxes)
      const validHeadshots = [];
      for (let k = 0; k < rawImages.length; k++) {
        const img = rawImages[k];
        if (!img) continue;
        
        // Try face detection - if we find a face, it's a headshot
        let cropResult = null;
        try {
          cropResult = await cropToFace(img);
        } catch (e) {
          // Ignore errors
        }
        
        if (cropResult) {
          validHeadshots.push({
            image: cropResult.image,
            faceX: cropResult.faceX,
            faceY: cropResult.faceY
          });
        } else {
          console.log(`[Cast] Filtered out rectangle ${k + 1} (no face - likely text)`);
        }
      }
      
      console.log(`[Cast] After filtering: ${validHeadshots.length} headshots for ${members.length} members`);
      
      for (let j = 0; j < members.length; j++) {
        const member = members[j];
        const headshot = validHeadshots[j] || null;
        const name = member.actorName || member.characterName || `Member ${j + 1}`;
        
        if (headshot) {
          console.log(`[Cast] ${name}: headshot at x=${(headshot.faceX*100).toFixed(0)}%, y=${(headshot.faceY*100).toFixed(0)}%`);
        } else {
          console.log(`[Cast] ${name}: no headshot available`);
        }
        
        allCast.push({
          actorName: member.actorName || '',
          characterName: member.characterName || '',
          role: member.role || '',
          age: member.age || null,
          dob: member.dob || null,
          isMinor: member.isMinor || false,
          hardIn: member.hardIn || null,
          hardOut: member.hardOut || null,
          unionStatus: member.unionStatus || null,
          image: headshot ? headshot.image : null,
          faceX: headshot ? headshot.faceX : 0.5,
          faceY: headshot ? headshot.faceY : 0.5
        });
      }
    }
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[Cast] Total: ${allCast.length} cast members (${elapsed}s)`);
    
    res.json({ cast: allCast });
    
  } catch (error) {
    console.error('[Cast] Error:', error);
    res.status(500).json({ error: error.message });
  } finally {
    try { await fs.rm(tempDir, { recursive: true, force: true }); } catch (e) {}
  }
});

// Crop image to face using face-api.js (primary) or Claude (fallback)
// Returns { image: base64, faceX: 0-1, faceY: 0-1 } or null
async function cropToFace(base64Image) {
  let faceCoords = null;
  
  // Try face-api.js first
  if (faceApiReady && faceapi) {
    try {
      const result = await detectFaceWithFaceApi(base64Image);
      if (result) {
        faceCoords = {
          x: result.x * 100,
          y: result.y * 100,
          size: result.radius * 100 * 2 // Convert radius to diameter percentage
        };
        console.log(`[Cast] face-api.js detected: x=${faceCoords.x.toFixed(0)}%, y=${faceCoords.y.toFixed(0)}%`);
      }
    } catch (err) {
      console.log('[Cast] face-api.js error:', err.message);
    }
  }
  
  // Fallback to Claude if face-api.js didn't work
  if (!faceCoords) {
    const client = getAnthropicClient();
    if (!client) return null;
    
    try {
      const response = await client.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 200,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: base64Image } },
            { type: 'text', text: `Find the face for a headshot crop. Return the CENTER point for cropping that shows the full head (forehead to chin). Position should be at eye level or slightly above, NOT on the nose.
Coordinates 0-100 where 0,0=top-left.
JSON only: {"x": 50, "y": 40, "size": 50}` }
          ]
        }]
      });
      
      const text = response.content?.[0]?.text || '';
      const jsonMatch = text.match(/\{[^}]+\}/);
      if (jsonMatch) {
        faceCoords = JSON.parse(jsonMatch[0]);
        console.log(`[Cast] Claude detected: x=${faceCoords.x}%, y=${faceCoords.y}%`);
      }
    } catch (err) {
      console.log('[Cast] Claude error:', err.message);
    }
  }
  
  if (!faceCoords) return null;
  
  try {
    // Decode the base64 image and crop to face
    const imageBuffer = Buffer.from(base64Image, 'base64');
    const metadata = await sharp(imageBuffer).metadata();
    
    const imgWidth = metadata.width;
    const imgHeight = metadata.height;
    
    // Calculate face position in pixels
    const faceX = (faceCoords.x / 100) * imgWidth;
    const faceY = (faceCoords.y / 100) * imgHeight;
    const faceSize = (faceCoords.size / 100) * imgWidth;
    
    // Make square crop around face center - zoom in tighter on face
    const cropSize = Math.min(Math.round(faceSize * 1.3), Math.min(imgWidth, imgHeight));
    const cropX = Math.max(0, Math.round(faceX - cropSize / 2));
    const cropY = Math.max(0, Math.round(faceY - cropSize / 2));
    
    // Ensure crop doesn't exceed image bounds
    const finalX = Math.min(cropX, imgWidth - cropSize);
    const finalY = Math.min(cropY, imgHeight - cropSize);
    const finalSize = Math.min(cropSize, imgWidth - finalX, imgHeight - finalY);
    
    if (finalSize < 50) return null;
    
    // Calculate where the face center is in the CROPPED image
    const faceXInCrop = (faceX - finalX) / finalSize;
    const faceYInCrop = (faceY - finalY) / finalSize;
    
    console.log(`[Cast] Cropping: face in crop at x=${(faceXInCrop*100).toFixed(0)}%, y=${(faceYInCrop*100).toFixed(0)}%`);
    
    // Crop and resize
    const croppedBuffer = await sharp(imageBuffer)
      .extract({ left: Math.max(0, finalX), top: Math.max(0, finalY), width: finalSize, height: finalSize })
      .resize(300, 300, { fit: 'cover' })
      .jpeg({ quality: 90 })
      .toBuffer();
    
    return {
      image: croppedBuffer.toString('base64'),
      faceX: Math.max(0, Math.min(1, faceXInCrop)),
      faceY: Math.max(0, Math.min(1, faceYInCrop))
    };
    
  } catch (e) {
    console.error('[Cast] cropToFace error:', e.message);
    return null;
  }
}

// AI function to extract cast info from page
async function extractCastText(imageBuffer) {
  const client = getAnthropicClient();
  if (!client) throw new Error('ANTHROPIC_API_KEY not set');
  
  const resized = await sharp(imageBuffer)
    .resize(1400, 1400, { fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 85 })
    .toBuffer();
  
  const response = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 4096,
    messages: [{
      role: 'user',
      content: [
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: resized.toString('base64') } },
        { type: 'text', text: `Extract cast/talent information from this page.

This is a cast sheet with headshot photos and names. Each name is positioned DIRECTLY BELOW its corresponding photo.

For each headshot/photo, identify:
1. NAME - The name directly below the photo (this is usually the character name or role)
2. Any additional info shown (age, DOB, union status, call times, etc.)

Return JSON in EXACT visual order - go row by row, left to right:
{
  "members": [
    {
      "actorName": "",
      "characterName": "NINA",
      "role": "",
      "age": null,
      "dob": null,
      "isMinor": false,
      "hardIn": null,
      "hardOut": null,
      "unionStatus": null
    }
  ]
}

RULES:
- Go row by row, left to right (top row first, then next row, etc.)
- The name BELOW each photo is typically the character name
- If only one name is shown, put it in characterName
- actorName: Real person's name (if shown separately)
- characterName: Character/role name shown below photo
- role: Additional description if any
- age: Numeric age if shown (e.g. 12). Set to null if not shown.
- dob: Date of birth string if shown (e.g. "01/15/2014"). Set to null if not shown.
- isMinor: true if person is under 18 or listed as minor/child. false otherwise.
- hardIn: Earliest start/call time if shown (e.g. "9:00 AM"). Set to null if not shown.
- hardOut: Latest end/wrap time if shown (e.g. "3:00 PM"). Set to null if not shown.
- unionStatus: "union" if SAG/AFTRA/union member, "non-union" if specified, null if not shown.
- Count must match number of photos exactly` }
      ]
    }]
  });
  
  const text = response.content[0].text;
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  try {
    return JSON.parse((jsonMatch ? jsonMatch[1] : text).trim());
  } catch (e) {
    console.error('[Cast] Text parse error');
    return { members: [] };
  }
}

// ============================================================================
// AUTO-TAG VISION ENDPOINT (single frame - legacy)
// ============================================================================

app.post('/api/auto-tag-vision', async (req, res) => {
  const { image, characters } = req.body;
  
  if (!image || !characters || !Array.isArray(characters)) {
    return res.status(400).json({ error: 'Missing image or characters array' });
  }
  
  if (!process.env.ANTHROPIC_API_KEY) {
    return res.status(500).json({ error: 'AI not configured' });
  }
  
  console.log('[AutoTagVision] Scanning for characters:', characters);
  
  try {
    const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 500,
      messages: [{
        role: 'user',
        content: [
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: 'image/jpeg',
              data: image
            }
          },
          {
            type: 'text',
            text: `This is a storyboard frame from a commercial shoot. Look for any text labels or character names in the image.

I'm looking for these character names: ${characters.join(', ')}

If you see any of these names written in the image (as labels, annotations, or dialogue), list ONLY the ones you find.

Respond with JSON only:
{"found": ["NAME1", "NAME2"]}

If none of those specific names appear in the image, respond:
{"found": []}`
          }
        ]
      }]
    });
    
    const text = response.content[0].text;
    console.log('[AutoTagVision] AI response:', text);
    
    // Parse JSON response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const result = JSON.parse(jsonMatch[0]);
      return res.json(result);
    }
    
    return res.json({ found: [] });
  } catch (error) {
    console.error('[AutoTagVision] Error:', error.message);
    return res.status(500).json({ error: error.message });
  }
});

// ============================================================================
// AUTO-TAG BATCH ENDPOINT (all frames at once for cross-referencing)
// ============================================================================

app.post('/api/auto-tag-batch', async (req, res) => {
  const { frames, characterRefs } = req.body;
  
  if (!frames || !Array.isArray(frames) || frames.length === 0) {
    return res.status(400).json({ error: 'Missing frames array' });
  }
  
  if (!characterRefs || !Array.isArray(characterRefs) || characterRefs.length === 0) {
    return res.status(400).json({ error: 'Missing characterRefs array' });
  }
  
  if (!process.env.ANTHROPIC_API_KEY) {
    return res.status(500).json({ error: 'AI not configured' });
  }
  
  const characters = characterRefs.map(c => c.characterName);
  const headshotsAvailable = characterRefs.filter(c => c.image).length;
  
  console.log('[AutoTagBatch] Processing', frames.length, 'rows for characters:', characters);
  console.log('[AutoTagBatch] Headshots available:', headshotsAvailable);
  
  try {
    const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    
    // Build content array
    const content = [];
    
    // FIRST: Add character reference headshots
    if (headshotsAvailable > 0) {
      content.push({
        type: 'text',
        text: '=== CHARACTER REFERENCE PHOTOS ===\nThese are photos of the actual actors. Use these to identify characters in the storyboard frames:'
      });
      
      for (const charRef of characterRefs) {
        if (charRef.image) {
          content.push({
            type: 'image',
            source: {
              type: 'base64',
              media_type: 'image/jpeg',
              data: charRef.image
            }
          });
          content.push({
            type: 'text',
            text: `This is ${charRef.characterName}`
          });
        }
      }
      
      content.push({
        type: 'text',
        text: '\n=== STORYBOARD FRAMES TO ANALYZE ===\n'
      });
    }
    
    // Add each row's images with description
    // Use very explicit labels so API doesn't confuse image sequence with row numbers
    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i];
      const images = frame.images || (frame.image ? [frame.image] : []);
      
      for (let j = 0; j < images.length; j++) {
        content.push({
          type: 'image',
          source: {
            type: 'base64',
            media_type: 'image/jpeg',
            data: images[j]
          }
        });
        
        if (images.length > 1) {
          content.push({
            type: 'text',
            text: `^^^ THIS IMAGE BELONGS TO ROW ${frame.rowNum} (image ${j + 1}/${images.length}) ${j === 0 ? frame.copyText || '' : ''}`
          });
        } else {
          content.push({
            type: 'text',
            text: `^^^ THIS IMAGE BELONGS TO ROW ${frame.rowNum} - ${frame.copyText || '(no description)'}`
          });
        }
      }
    }
    
    // Add the analysis prompt
    content.push({
      type: 'text',
      text: `You are analyzing storyboard frames from a commercial shoot. Your task is to identify characters VISUALLY PRESENT in each frame.

CHARACTERS TO IDENTIFY: ${characters.join(', ')}
${headshotsAvailable > 0 ? '\nYou have reference photos of each character above. Match the storyboard drawings to these real faces - pay attention to gender, hair, and build.' : ''}

=== IMPORTANT: ROW NUMBERS ===

Each image above is labeled with its ROW NUMBER (e.g., "ROW 1", "ROW 2", etc.).
Some rows have MULTIPLE images - combine all characters from all images in that row.
Use the ROW NUMBERS from the labels - do NOT number images sequentially yourself.

There are exactly ${frames.length} rows to analyze. Your response must have exactly ${frames.length} assignments.

=== CRITICAL RULE ===

COUNT BODIES FIRST. The number of characters you tag MUST EQUAL the number of human figures drawn.

If you see 1 person → tag exactly 1 character
If you see 2 people → tag exactly 2 characters  
If you see 3 people → tag exactly 3 characters

NEVER tag more characters than bodies visible. This is the most important rule.

=== ANALYSIS STEPS ===

For each ROW (using the row number from the label):
1. Count human figures DRAWN in the image(s) - write this number down
2. Identify each figure by matching to reference photos
3. Verify: does your character count match your body count? If not, fix it.

STRICT RULES:
- Only tag characters you can SEE drawn
- Descriptions often mention characters NOT in frame - ignore the text, trust your eyes
- ${headshotsAvailable > 0 ? 'Use reference photos to distinguish similar characters' : 'Build profiles from establishing shots'}

Respond with JSON:
{
  "assignments": [
    {"rowNum": "1", "bodyCount": 1, "characters": ["RICK"], "reasoning": "1 male figure in doorway"},
    {"rowNum": "2", "bodyCount": 2, "characters": ["RICK", "TANYA"], "reasoning": "2 figures - male + female entering"},
    ...
  ]
}

VALIDATION: 
- Use the ROW NUMBERS from the image labels (ROW 1, ROW 2, etc.)
- You must have exactly ${frames.length} assignments
- For each row, characters.length MUST equal bodyCount`
    });
    
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 8000,
      messages: [{
        role: 'user',
        content
      }]
    });
    
    const text = response.content[0].text;
    console.log('[AutoTagBatch] AI response length:', text.length);
    console.log('[AutoTagBatch] AI response preview:', text.substring(0, 500));
    
    // Parse JSON response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try {
        const result = JSON.parse(jsonMatch[0]);
        
        // Log character profiles if present
        if (result.characterProfiles) {
          console.log('[AutoTagBatch] Character profiles:', Object.keys(result.characterProfiles));
          for (const [char, profile] of Object.entries(result.characterProfiles)) {
            console.log(`  ${char}: ${profile.substring(0, 100)}...`);
          }
        }
        
        // Add rowId mapping back
        if (result.assignments) {
          result.assignments = result.assignments.map(a => {
            const frame = frames.find(f => f.rowNum === a.rowNum || f.rowNum === String(a.rowNum));
            if (a.reasoning) {
              console.log(`[AutoTagBatch] Row ${a.rowNum}: ${a.characters.join(', ')} - ${a.reasoning}`);
            }
            return {
              ...a,
              rowId: frame?.rowId || null
            };
          });
        }
        
        console.log('[AutoTagBatch] Parsed', result.assignments?.length, 'assignments');
        return res.json(result);
      } catch (parseErr) {
        console.error('[AutoTagBatch] JSON parse error:', parseErr);
        console.error('[AutoTagBatch] Raw text:', text.substring(0, 500));
        return res.status(500).json({ error: 'Failed to parse AI response' });
      }
    }
    
    return res.json({ assignments: [] });
  } catch (error) {
    console.error('[AutoTagBatch] Error:', error.message);
    return res.status(500).json({ error: error.message });
  }
});

// Face detection endpoint - uses face-api.js for accurate landmark detection
app.post('/api/detect-face', async (req, res) => {
  try {
    const { image } = req.body; // base64 image
    
    if (!image) {
      return res.status(400).json({ error: 'Image required' });
    }
    
    // Try face-api.js first (most accurate)
    if (faceApiReady && faceapi) {
      try {
        const result = await detectFaceWithFaceApi(image);
        if (result) {
          console.log('[FaceDetect] face-api.js detected:', result);
          return res.json({ ...result, method: 'faceapi' });
        }
      } catch (err) {
        console.log('[FaceDetect] face-api.js error:', err.message);
      }
    }
    
    // Fallback to Claude
    const client = getAnthropicClient();
    if (client) {
      try {
        console.log('[FaceDetect] Falling back to Claude...');
        const response = await client.messages.create({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 200,
          messages: [{
            role: 'user',
            content: [
              { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: image } },
              { type: 'text', text: `Find the person's face. Return the nose bridge position (center of face).
Coordinates 0-1 where 0,0=top-left.
Reply ONLY with JSON: {"x": 0.5, "y": 0.35, "radius": 0.4}` }
            ]
          }]
        });
        
        const text = response.content?.[0]?.text || '';
        const jsonMatch = text.match(/\{[^}]+\}/);
        if (jsonMatch) {
          const result = JSON.parse(jsonMatch[0]);
          console.log('[FaceDetect] Claude detected:', result);
          return res.json({ ...result, method: 'claude' });
        }
      } catch (err) {
        console.log('[FaceDetect] Claude error:', err.message);
      }
    }
    
    // Final fallback
    console.log('[FaceDetect] Using heuristic fallback');
    return res.json({ x: 0.5, y: 0.4, radius: 0.45, method: 'heuristic' });
    
  } catch (error) {
    console.error('[FaceDetect] Error:', error.message);
    return res.json({ x: 0.5, y: 0.4, radius: 0.45, method: 'heuristic' });
  }
});

// Face detection using face-api.js with 68-point landmarks
async function detectFaceWithFaceApi(base64Image) {
  if (!faceApiReady || !faceapi) return null;
  
  try {
    const canvas = await import('canvas');
    const { Canvas, Image, loadImage, createCanvas } = canvas.default || canvas;
    
    // Load image from base64
    const imgBuffer = Buffer.from(base64Image, 'base64');
    let img = await loadImage(imgBuffer);
    
    console.log(`[FaceAPI] Input image size: ${img.width}x${img.height}`);
    
    // Upscale small images for better detection (face-api needs ~100px minimum face size)
    const minDimension = Math.min(img.width, img.height);
    if (minDimension < 400) {
      const scale = 400 / minDimension;
      const newWidth = Math.round(img.width * scale);
      const newHeight = Math.round(img.height * scale);
      
      const upscaleCanvas = createCanvas(newWidth, newHeight);
      const ctx = upscaleCanvas.getContext('2d');
      ctx.drawImage(img, 0, 0, newWidth, newHeight);
      img = upscaleCanvas;
      console.log(`[FaceAPI] Upscaled to: ${newWidth}x${newHeight}`);
    }
    
    // Detect face with landmarks
    const detection = await faceapi
      .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks(true); // true = use tiny model
    
    if (!detection) {
      console.log('[FaceAPI] No face detected');
      return null;
    }
    
    const landmarks = detection.landmarks;
    const imgWidth = img.width;
    const imgHeight = img.height;
    
    // Get nose position (points 27-30 are the nose bridge)
    const noseBridge = landmarks.getNose();
    
    // Get eye positions for better centering
    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();
    
    // Calculate eye center
    const leftEyeCenter = {
      x: leftEye.reduce((sum, p) => sum + p.x, 0) / leftEye.length,
      y: leftEye.reduce((sum, p) => sum + p.y, 0) / leftEye.length
    };
    const rightEyeCenter = {
      x: rightEye.reduce((sum, p) => sum + p.x, 0) / rightEye.length,
      y: rightEye.reduce((sum, p) => sum + p.y, 0) / rightEye.length
    };
    
    // Face center X is between the eyes
    const faceCenterX = (leftEyeCenter.x + rightEyeCenter.x) / 2;
    
    // Face center Y should be at eye level for good headshot framing
    // Eyes are typically in the upper third of a good headshot
    const eyeLevel = (leftEyeCenter.y + rightEyeCenter.y) / 2;
    const faceCenterY = eyeLevel;
    
    // Calculate face size based on bounding box
    // Use larger radius to show full head + some shoulders
    const box = detection.detection.box;
    const faceSize = Math.max(box.width, box.height);
    const radius = (faceSize / Math.min(imgWidth, imgHeight)) * 0.6;
    
    console.log(`[FaceAPI] Landmarks: eyes at y=${(eyeLevel / imgHeight * 100).toFixed(0)}%`);
    
    return {
      x: faceCenterX / imgWidth,
      y: faceCenterY / imgHeight,
      radius: Math.min(0.5, Math.max(0.3, radius))
    };
    
  } catch (err) {
    console.error('[FaceAPI] Detection error:', err.message);
    return null;
  }
}

// ============================================================================
// FOLD: EXTRACT CONSTRAINTS FROM DOCUMENT TEXT
// ============================================================================

app.post('/api/extract-constraints', async (req, res) => {
  try {
    const { text } = req.body;
    
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Missing or invalid text field' });
    }
    
    const client = getAnthropicClient();
    if (!client) {
      return res.status(500).json({ error: 'ANTHROPIC_API_KEY not set' });
    }
    
    console.log(`[Constraints] Analyzing ${text.length} chars of document text`);
    const startTime = Date.now();
    
    const prompt = `Analyze these production documents and extract ALL scheduling constraints, requirements, and important information. Look for:

1. TIME CONSTRAINTS: Hard in times, hard out times, call times, wrap times, meal breaks, turnaround requirements, overtime limits
2. TALENT CONSTRAINTS: Actor availability, fitting times, pickup times, drop-off times, consecutive work day limits
3. MINOR/CHILD PERFORMERS: Ages, dates of birth (DOB), minor status, school day requirements, work hour restrictions for minors
4. LOCATION CONSTRAINTS: Permit windows, noise restrictions, parking limitations, load-in/load-out times
5. EQUIPMENT CONSTRAINTS: Rental pickup/return times, special equipment availability windows
6. UNION STATUS: SAG-AFTRA, IATSE, or other union memberships and their associated rules
7. OTHER CONSTRAINTS: Weather dependencies, daylight requirements, union rules, safety requirements

IMPORTANT: For each person mentioned, always extract:
- Their age or date of birth if mentioned anywhere in the document
- Whether they are a minor (under 18)
- Any hard in (earliest start) or hard out (latest end) times
- Union or non-union status

For each constraint found, extract:
- The specific constraint or requirement
- Who/what it applies to (use the person's name exactly as written)
- The time/date if specified
- The source document

Return as JSON array with format:
[
  {
    "type": "time|talent|minor|location|equipment|union|other",
    "text": "Clear description of the constraint",
    "who": "Person/thing it applies to (if applicable)",
    "when": "Time/date (if specified)",
    "source": "Document name"
  }
]

If a document is empty, unreadable, or contains no scheduling constraints, return an empty array []. Do NOT create constraints about documents being empty or unreadable.

Documents to analyze:
${text.substring(0, 50000)}`;

    const response = await client.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      messages: [{ role: 'user', content: prompt }]
    });
    
    const content = response.content?.[0]?.text || '';
    
    // Parse JSON from response
    const jsonMatch = content.match(/\[[\s\S]*\]/);
    let constraints = [];
    if (jsonMatch) {
      try {
        constraints = JSON.parse(jsonMatch[0]);
      } catch (e) {
        console.error('[Constraints] JSON parse error:', e.message);
      }
    }
    
    // Filter out junk/error constraints
    constraints = constraints.filter(c => {
      const lt = (c.text || '').toLowerCase();
      if (lt.includes('appears to be empty') || lt.includes('not accessible') ||
          lt.includes('could not extract') || lt.includes('no constraints found') ||
          lt.includes('unable to read') || lt.includes('no relevant') ||
          (c.who === 'N/A' && c.when === 'N/A' && c.type === 'other')) {
        console.log('[Constraints] Filtered junk:', c.text);
        return false;
      }
      return true;
    });
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[Constraints] Extracted ${constraints.length} constraints (${elapsed}s)`);
    
    res.json({ constraints });
    
  } catch (error) {
    console.error('[Constraints] Error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server on port ${PORT}`);
  console.log(`Storyboard: ${process.env.ANTHROPIC_API_KEY ? 'enabled' : 'disabled'}`);
  console.log(`Face detection: ${faceApiReady ? 'face-api.js' : (process.env.ANTHROPIC_API_KEY ? 'claude' : 'disabled')}`);
});
