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
    const allowed = ['application/pdf', 'image/jpeg', 'image/png', 'image/webp',
      'application/vnd.openxmlformats-officedocument.presentationml.presentation', // .pptx
      'application/vnd.ms-powerpoint', // .ppt
      'application/x-iwork-keynote-sffkey', // .key (newer)
      'application/vnd.apple.keynote', // .key (older)
    ];
    // Allow octet-stream only for presentation file extensions (.key, .pptx)
    if (file.mimetype === 'application/octet-stream') {
      const ext = (file.originalname || '').toLowerCase();
      cb(null, ext.endsWith('.key') || ext.endsWith('.pptx') || ext.endsWith('.ppt'));
    } else {
      cb(null, allowed.includes(file.mimetype));
    }
  }
});

let anthropic = null;
function getAnthropicClient() {
  if (!anthropic && process.env.ANTHROPIC_API_KEY) {
    anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  }
  return anthropic;
}

/**
 * Retry wrapper for Anthropic API calls with exponential backoff.
 * Retries on 529 (overloaded) and 500+ errors.
 */
async function apiCallWithRetry(fn, maxRetries = 5) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const status = err?.status || 0;
      const retryable = status === 529 || status >= 500;
      if (!retryable || attempt === maxRetries) throw err;
      const delay = 2000 * Math.pow(2, attempt) + Math.random() * 1000;
      console.log(`[API Retry] ${status} error, attempt ${attempt + 1}/${maxRetries}, waiting ${Math.round(delay)}ms`);
      await new Promise(r => setTimeout(r, delay));
    }
  }
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
          console.log(`[Storyboard] Found ${result.count} rectangles (mode: ${result.mode || 'unknown'})`);
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
 * Extract panels directly from PDF structure using PyMuPDF (pdf_panels.py).
 * This is the FIRST option — reads image placement coordinates from the PDF,
 * giving pixel-perfect crops without needing to render and re-detect.
 * Returns per-page results with images and captions, or null if extraction fails.
 */
async function extractPanelsFromPdfStructure(pdfBuffer, tempDir) {
  const pdfPath = path.join(tempDir, 'input.pdf');
  await fs.writeFile(pdfPath, pdfBuffer);

  // Get page count first using a quick PyMuPDF call
  const pageCount = await new Promise((resolve) => {
    const script = path.join(__dirname, 'pdf_panels.py');
    const tryPython = (cmd) => {
      // Quick page-count probe: extract page 1 just to see if fitz works
      const proc = spawn(cmd, [script, pdfPath, '1', '100', '50']);
      let stdout = '';
      let stderr = '';
      proc.stdout.on('data', d => stdout += d);
      proc.stderr.on('data', d => stderr += d);
      proc.on('close', code => {
        if (code !== 0) {
          if (cmd === 'python3') { tryPython('python'); return; }
          resolve(0);
          return;
        }
        try {
          const result = JSON.parse(stdout);
          // Use a separate call to get actual page count
          resolve(result); // We'll get page count differently
        } catch (e) { resolve(0); }
      });
      proc.on('error', () => {
        if (cmd === 'python3') { tryPython('python'); return; }
        resolve(0);
      });
    };
    tryPython('python3');
  });

  // Get actual page count via PyMuPDF
  const totalPages = await new Promise((resolve) => {
    const tryPython = (cmd) => {
      const proc = spawn(cmd, ['-c', `import fitz; doc = fitz.open("${pdfPath}"); print(len(doc)); doc.close()`]);
      let stdout = '';
      proc.stdout.on('data', d => stdout += d);
      proc.on('close', code => {
        if (code !== 0) {
          if (cmd === 'python3') { tryPython('python'); return; }
          resolve(0);
          return;
        }
        resolve(parseInt(stdout.trim()) || 0);
      });
      proc.on('error', () => {
        if (cmd === 'python3') { tryPython('python'); return; }
        resolve(0);
      });
    };
    tryPython('python3');
  });

  if (totalPages < 1) {
    console.log('[Storyboard] pdf_panels.py: could not open PDF');
    return null;
  }

  console.log(`[Storyboard] pdf_panels.py: PDF has ${totalPages} pages`);

  // Extract panels from each page
  const results = [];
  for (let pageNum = 1; pageNum <= totalPages; pageNum++) {
    const pageResult = await new Promise((resolve) => {
      const script = path.join(__dirname, 'pdf_panels.py');
      const tryPython = (cmd) => {
        const proc = spawn(cmd, [script, pdfPath, String(pageNum), '80', '40']);
        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', d => stdout += d);
        proc.stderr.on('data', d => stderr += d);
        proc.on('close', code => {
          if (stderr) console.log(`[Storyboard] pdf_panels.py page ${pageNum}: ${stderr.trim()}`);
          if (code !== 0) {
            if (cmd === 'python3') { tryPython('python'); return; }
            resolve(null);
            return;
          }
          try {
            resolve(JSON.parse(stdout));
          } catch (e) {
            resolve(null);
          }
        });
        proc.on('error', () => {
          if (cmd === 'python3') { tryPython('python'); return; }
          resolve(null);
        });
      };
      tryPython('python3');
    });

    results.push({ pageNum, data: pageResult });
  }

  // Check if we got meaningful results (at least some pages with panels)
  const pagesWithPanels = results.filter(r => r.data && r.data.count > 0);
  if (pagesWithPanels.length === 0) {
    console.log('[Storyboard] pdf_panels.py: no panels found on any page');
    return null;
  }

  console.log(`[Storyboard] pdf_panels.py: found panels on ${pagesWithPanels.length}/${totalPages} pages`);
  return { totalPages, pages: results, pdfPath };
}

/**
 * Check if a file is a PPTX/KEY presentation format
 */
function isPresentationFile(mimetype, originalname) {
  const presentationMimes = [
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'application/vnd.ms-powerpoint',
    'application/x-iwork-keynote-sffkey',
    'application/vnd.apple.keynote',
  ];
  if (presentationMimes.includes(mimetype)) return true;
  // Fallback: check extension for octet-stream uploads
  const ext = path.extname(originalname || '').toLowerCase();
  return ['.pptx', '.ppt', '.key'].includes(ext);
}

/**
 * Convert PPTX/KEY to PDF using LibreOffice, then process as PDF
 * Returns { pdfBuffer, slideCount }
 */
async function convertPresentationToPdf(fileBuffer, originalname, tempDir) {
  const ext = path.extname(originalname || '.pptx').toLowerCase();
  const inputPath = path.join(tempDir, `presentation${ext}`);
  await fs.writeFile(inputPath, fileBuffer);

  console.log(`[Storyboard] Converting ${ext} to PDF via LibreOffice...`);

  return new Promise((resolve, reject) => {
    const proc = spawn('libreoffice', [
      '--headless', '--convert-to', 'pdf',
      '--outdir', tempDir,
      inputPath
    ], { timeout: 120000 });

    let stderr = '';
    proc.stderr.on('data', d => stderr += d.toString());

    proc.on('close', async (code) => {
      if (code !== 0) {
        return reject(new Error(`LibreOffice conversion failed (code ${code}): ${stderr}`));
      }

      const pdfPath = path.join(tempDir, `presentation.pdf`);
      try {
        const pdfBuffer = await fs.readFile(pdfPath);
        console.log(`[Storyboard] LibreOffice conversion complete: ${(pdfBuffer.length / 1024 / 1024).toFixed(1)}MB PDF`);
        resolve({ pdfBuffer });
      } catch (e) {
        reject(new Error(`LibreOffice produced no PDF output: ${e.message}`));
      }
    });

    proc.on('error', reject);
  });
}

/**
 * Extract panel images from PPTX using pptx_panels.py (like pdf_panels.py for PDFs)
 */
async function extractPanelsFromPptxStructure(fileBuffer, tempDir) {
  const pptxPath = path.join(tempDir, 'source.pptx');
  await fs.writeFile(pptxPath, fileBuffer);

  // Get slide count
  const slideCountResult = await new Promise((resolve, reject) => {
    const proc = spawn('python3', [path.join(__dirname, 'pptx_panels.py'), pptxPath]);
    let stdout = '', stderr = '';
    proc.stdout.on('data', d => stdout += d.toString());
    proc.stderr.on('data', d => { stderr += d.toString(); console.log(d.toString().trim()); });
    proc.on('close', (code) => {
      try { resolve(JSON.parse(stdout)); }
      catch { reject(new Error(`pptx_panels.py slide count failed: ${stderr}`)); }
    });
    proc.on('error', reject);
  });

  if (!slideCountResult.slideCount) {
    console.log('[Storyboard] pptx_panels.py could not read slide count');
    return null;
  }

  const totalPages = slideCountResult.slideCount;
  console.log(`[Storyboard] PPTX has ${totalPages} slides`);

  // Extract panels from each slide
  const pages = [];
  for (let i = 1; i <= totalPages; i++) {
    const result = await new Promise((resolve, reject) => {
      const proc = spawn('python3', [path.join(__dirname, 'pptx_panels.py'), pptxPath, String(i)]);
      let stdout = '', stderr = '';
      proc.stdout.on('data', d => stdout += d.toString());
      proc.stderr.on('data', d => { stderr += d.toString(); console.log(d.toString().trim()); });
      proc.on('close', (code) => {
        try { resolve(JSON.parse(stdout)); }
        catch { resolve({ count: 0, panels: [] }); }
      });
      proc.on('error', () => resolve({ count: 0, panels: [] }));
    });

    // Convert pptx_panels.py format to match pdf_panels.py format expected by server
    const images = (result.panels || []).map(p => ({
      x: p.x, y: p.y, width: p.width, height: p.height,
      image: p.image,
      caption: p.caption || ''
    }));

    pages.push({
      pageNum: i,
      data: {
        count: images.length,
        images,
        pageWidth: 960,  // Standard PPTX width in points (10in * 96)
        pageHeight: 540,  // Standard PPTX height
        title: result.title || null,
        mode: 'pptx_structure'
      }
    });
  }

  const pagesWithPanels = pages.filter(p => p.data.count > 0);
  console.log(`[Storyboard] pptx_panels.py: found panels on ${pagesWithPanels.length}/${totalPages} slides`);
  return { totalPages, pages, pptxPath };
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
    await page.setViewport({ width: 1400, height: 1800, deviceScaleFactor: 2 });
    
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
          const vp = pg.getViewport({scale: 3});
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
        .resize(2400, 2400, { fit: 'inside', withoutEnlargement: true })
        .jpeg({ quality: 90, progressive: true })
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

STEP 2 - BOARD TYPE:
Classify the drawing panels:
- "hand_drawn": panels contain hand-drawn illustrations, sketches, pencil/ink drawings, or animatic-style artwork
- "photo": panels contain photographs, rendered images, composited images, or photographic reference

STEP 3 - GRID LAYOUT:
Identify the grid structure. Read frames LEFT-TO-RIGHT, then TOP-TO-BOTTOM.

STEP 4 - FRAME NUMBERS:
- If frames have visible numbers (1, 2, 1A, 1B, 1.1, 1.2, 3.1, etc.), use those EXACTLY as written
- Decimal numbers like 1.1, 1.2, 2.1, 3.1 are common in storyboards — preserve the decimal format exactly
- If NO visible numbers, number sequentially: 1, 2, 3, 4...

STEP 5 - EXTRACT:
For each frame, identify which IMAGE/PANEL it belongs to and estimate that panel's center position on the page as a percentage (0-100).

Return JSON:
{
  "spotName": "EXACT TITLE FROM PAGE" or null,
  "boardType": "hand_drawn" or "photo",
  "gridLayout": "2x3",
  "hasVisibleNumbers": true/false,
  "frames": [
    {
      "frameNumber": "1.1",
      "description": "Action/direction text",
      "dialog": "CHARACTER: Spoken lines...",
      "panelX": 25,
      "panelY": 35
    }
  ]
}

panelX/panelY = approximate CENTER of the IMAGE/PANEL (not the text) as percentage of page width/height (0=left/top, 100=right/bottom).

RULES:
- spotName: ALWAYS extract the bold title at top of page - this identifies which commercial/spot
- boardType: classify based on the DRAWING PANELS content, not the page layout
- description: action/camera direction text
- dialog: spoken lines with character prefix
- CRITICAL: Count ALL image panels on the page. The number of frames you return MUST match the total number of IMAGE PANELS in the grid layout (e.g. 2x3 = 6 frames, 4+5+4 = 13 frames). Do NOT skip any panels.
- Include frames even if they have no text (use empty strings for description/dialog)
- Text may appear below, beside, or near its panel — pair each text block with its nearest image` }
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

/**
 * Detect panels from a pre-thresholded mask using OpenCV contour detection.
 * Returns array of base64 images cropped from the ORIGINAL full-res image, or null if detection fails.
 */
async function detectPanelsFromMask(maskBuffer, originalImageBuffer, expectedCount, boardType = 'photo') {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'mask-'));
  const maskPath = path.join(tmpDir, 'mask.jpg');

  try {
    await fs.writeFile(maskPath, maskBuffer);

    // Get mask dimensions for coordinate scaling
    const maskMeta = await sharp(maskBuffer).metadata();
    const origMeta = await sharp(originalImageBuffer).metadata();
    const scaleX = origMeta.width / maskMeta.width;
    const scaleY = origMeta.height / maskMeta.height;

    // Run Python — just get rectangles, no cropping
    const result = await new Promise((resolve) => {
      const script = path.join(__dirname, 'frame_detector.py');
      const args = [script, maskPath, 'mask', String(expectedCount)];
      
      const tryPython = (cmd) => {
        const proc = spawn(cmd, args);
        let stdout = '';
        let stderr = '';
        
        proc.stdout.on('data', d => stdout += d);
        proc.stderr.on('data', d => stderr += d);
        
        proc.on('close', code => {
          if (stderr) console.log(`[Storyboard] Mask-OpenCV debug: ${stderr.trim()}`);
          if (code !== 0) {
            if (cmd === 'python3') {
              tryPython('python');
              return;
            }
            resolve(null);
            return;
          }
          try {
            resolve(JSON.parse(stdout));
          } catch (e) {
            resolve(null);
          }
        });
        
        proc.on('error', () => {
          if (cmd === 'python3') {
            tryPython('python');
            return;
          }
          resolve(null);
        });
      };
      
      tryPython('python3');
    });
    
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});

    if (!result || !result.rectangles || result.count < 1) return null;

    console.log(`[Storyboard] Mask-OpenCV: found ${result.count} panels`);

    // Crop from full-resolution original with scaled coordinates
    const images = [];
    const panelCentroids = []; // Normalized 0-1 centroids for proximity matching
    for (const rect of result.rectangles) {
      try {
        const x = Math.max(0, Math.round(rect.x * scaleX));
        const y = Math.max(0, Math.round(rect.y * scaleY));
        const w = Math.min(Math.round(rect.width * scaleX), origMeta.width - x);
        const h = Math.min(Math.round(rect.height * scaleY), origMeta.height - y);

        // Store normalized centroid (0-1 relative to page)
        panelCentroids.push({
          cx: (rect.x + rect.width / 2) / maskMeta.width,
          cy: (rect.y + rect.height / 2) / maskMeta.height
        });

        if (w < 20 || h < 20) {
          images.push(null);
          continue;
        }

        // 3px inset at original scale to trim border edges
        const inset = Math.round(3 * scaleX);
        const cropped = await sharp(originalImageBuffer)
          .extract({
            left: x + inset,
            top: y + inset,
            width: Math.max(10, w - inset * 2),
            height: Math.max(10, h - inset * 2)
          })
          .jpeg({ quality: 85 })
          .toBuffer();

        // Run per-panel text erasure (trim caption zone + OCR cleanup)
        // Skip OCR for hand-drawn boards (Tesseract misreads artwork)
        if (boardType === 'hand_drawn') {
          images.push(cropped.toString('base64'));
        } else {
          const cleaned = await eraseTextFromCrop(cropped, images.length + 1);
          images.push(cleaned || cropped.toString('base64'));
        }
      } catch (e) {
        console.error('[Storyboard] Mask-OpenCV crop error:', e.message);
        images.push(null);
      }
    }

    return { images, panelCentroids };

  } catch (e) {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
    console.error('[Storyboard] Mask-OpenCV error:', e.message);
    return null;
  }
}

/**
 * Vision panel detection — hybrid approach.
 * 1. Generate binary mask from the image
 * 2. Try OpenCV contour detection on the mask (exact pixel edges)
 * 3. If that fails, fall back to Claude Vision with dual-image (approximate)
 */
async function detectPanelsWithVision(imageBuffer, expectedCount, boardType = 'photo') {
  // Get original dimensions for cropping later
  const originalMeta = await sharp(imageBuffer).metadata();
  const origW = originalMeta.width;
  const origH = originalMeta.height;

  // Resize original for processing (2000px for better Vision bbox precision)
  const resized = await sharp(imageBuffer)
    .resize(2000, 2000, { fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 85 })
    .toBuffer();

  const resizedMeta = await sharp(resized).metadata();
  const imgW = resizedMeta.width;
  const imgH = resizedMeta.height;

  // Detect background color dynamically by sampling at multiple depths from edge
  const grayBuf = await sharp(imageBuffer)
    .resize(2000, 2000, { fit: 'inside', withoutEnlargement: true })
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  const grayData = grayBuf.data;
  const gW = grayBuf.info.width;
  const gH = grayBuf.info.height;
  
  // Sample pixels at 0%, 5%, 10%, 15% depth from each edge
  const perimeterValues = [];
  for (const depthPct of [0, 5, 10, 15]) {
    const offX = Math.round(gW * depthPct / 100);
    const offY = Math.round(gH * depthPct / 100);
    for (let x = offX; x < gW - offX; x += 5) {
      perimeterValues.push(grayData[offY * gW + x]);
      perimeterValues.push(grayData[(gH - 1 - offY) * gW + x]);
    }
    for (let y = offY; y < gH - offY; y += 5) {
      perimeterValues.push(grayData[y * gW + offX]);
      perimeterValues.push(grayData[y * gW + (gW - 1 - offX)]);
    }
  }
  
  // Bucket into groups of 5, find background color
  const buckets = {};
  for (const v of perimeterValues) {
    if (v < 150) continue;
    const bucket = Math.round(v / 5) * 5;
    buckets[bucket] = (buckets[bucket] || 0) + 1;
  }
  const sortedBuckets = Object.entries(buckets).sort((a, b) => b[1] - a[1]);
  
  let bgValue = 245;
  if (sortedBuckets.length > 0) {
    const topValue = parseInt(sortedBuckets[0][0]);
    const topCount = sortedBuckets[0][1];
    if (topValue >= 250 && sortedBuckets.length > 1) {
      const secondValue = parseInt(sortedBuckets[1][0]);
      const secondCount = sortedBuckets[1][1];
      if (secondValue >= 150 && secondCount > topCount * 0.1) {
        bgValue = secondValue;
      } else {
        bgValue = topValue;
      }
    } else {
      bgValue = topValue;
    }
  }
  const dynamicThreshold = Math.max(100, bgValue - 15);
  
  console.log(`[Storyboard] Vision: detected background=${bgValue}, using threshold=${dynamicThreshold}`);

  // Generate binary mask
  const mask = await sharp(imageBuffer)
    .resize(2000, 2000, { fit: 'inside', withoutEnlargement: true })
    .grayscale()
    .threshold(dynamicThreshold)
    .jpeg({ quality: 90 })
    .toBuffer();

  // === HYBRID STEP: Try OpenCV on mask first ===
  const maskResult = await detectPanelsFromMask(mask, imageBuffer, expectedCount, boardType);
  const maskPanels = maskResult ? maskResult.images : null;
  const maskCentroids = maskResult ? maskResult.panelCentroids : null;

  if (maskPanels && maskPanels.length >= Math.max(1, Math.ceil(expectedCount * 0.7))) {
    console.log(`[Storyboard] Mask-OpenCV succeeded: ${maskPanels.length} panels (expected ~${expectedCount})`);
    return { images: maskPanels, panelCentroids: maskCentroids };
  }

  console.log(`[Storyboard] Mask-OpenCV insufficient (found ${maskPanels ? maskPanels.length : 0}, expected ~${expectedCount}) — falling back to Vision`);

  // === FALLBACK: Claude Vision with dual-image ===
  const client = getAnthropicClient();
  if (!client) return { images: [], panelCentroids: [] };

  const scaleX = origW / imgW;
  const scaleY = origH / imgH;

  console.log(`[Storyboard] Vision: sending ${imgW}x${imgH} original + mask (expecting ~${expectedCount} panels)`);

  const response = await apiCallWithRetry(() => client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 2048,
    temperature: 0,
    messages: [{
      role: 'user',
      content: [
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: resized.toString('base64') } },
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: mask.toString('base64') } },
        { type: 'text', text: `Above are two versions of the same storyboard page (${imgW}x${imgH} pixels):
1. The ORIGINAL storyboard page
2. A HIGH-CONTRAST MASK where dark content appears black and background is white

Use BOTH images to identify the drawing panels. The original shows the actual content and any border lines. The mask helps distinguish panels from background on photo-heavy boards.

There are approximately ${expectedCount} panels arranged in rows.

Find each individual DRAWING PANEL — the rectangular areas containing artwork, photos, or sketches. Each panel is a SEPARATE rectangle. Do NOT merge adjacent panels even if they touch.

CRITICAL: Return TIGHT bounding boxes around the ILLUSTRATION/ARTWORK ONLY.
- Do NOT include text captions, descriptions, or dialogue that appears BELOW, BESIDE, or ABOVE each panel.
- Do NOT include frame number labels (e.g. "1", "2A", "FR 3").
- The bounding box should END where the drawing/photo ends, NOT where the text below it ends.
- Caption text is typically printed in a separate zone under each panel with a small gap — exclude that zone entirely.
- If there is a visible border around the illustration, the bounding box should match the border edges.

Return ONLY JSON:
{
  "panels": [
    {"x": 50, "y": 100, "w": 300, "h": 220},
    {"x": 380, "y": 100, "w": 300, "h": 220}
  ]
}

x,y = top-left corner. w,h = width and height in pixels.
Read LEFT-TO-RIGHT, then TOP-TO-BOTTOM.` }
      ]
    }]
  }));

  const text = response.content[0]?.text || '';
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) || [null, text];

  let panels;
  try {
    const parsed = JSON.parse(jsonMatch[1].trim());
    panels = parsed.panels || [];
  } catch (e) {
    const objMatch = text.match(/\{[\s\S]*"panels"[\s\S]*\}/);
    if (objMatch) {
      try {
        panels = JSON.parse(objMatch[0]).panels || [];
      } catch (_) {
        console.error('[Storyboard] Vision: failed to parse response');
        return { images: [], panelCentroids: [] };
      }
    } else {
      return { images: [], panelCentroids: [] };
    }
  }

  console.log(`[Storyboard] Vision: found ${panels.length} panels (dual-image)`);

  // Compute normalized centroids from Vision-reported panel positions
  const panelCentroids = panels.map(p => ({
    cx: (p.x + p.w / 2) / imgW,
    cy: (p.y + p.h / 2) / imgH
  }));

  // Crop from original resolution image
  const images = [];
  for (let i = 0; i < panels.length; i++) {
    const p = panels[i];
    try {
      // Apply 3% margin expansion to compensate for Vision coordinate imprecision
      const marginX = Math.round(p.w * 0.03);
      const marginY = Math.round(p.h * 0.03);
      const ex = Math.max(0, p.x - marginX);
      const ey = Math.max(0, p.y - marginY);
      const ew = Math.min(p.w + marginX * 2, imgW - ex);
      const eh = Math.min(p.h + marginY * 2, imgH - ey);

      const x = Math.max(0, Math.round(ex * scaleX));
      const y = Math.max(0, Math.round(ey * scaleY));
      const w = Math.min(Math.round(ew * scaleX), origW - x);
      const h = Math.min(Math.round(eh * scaleY), origH - y);

      if (w < 20 || h < 20) {
        images.push(null);
        continue;
      }

      const cropped = await sharp(imageBuffer)
        .extract({ left: x, top: y, width: w, height: h })
        .jpeg({ quality: 85 })
        .toBuffer();

      // Run per-panel text erasure — but skip OCR for hand-drawn boards
      // (Tesseract misreads ink strokes as text, damaging artwork)
      if (boardType === 'hand_drawn') {
        images.push(cropped.toString('base64'));
      } else {
        const cleaned = await eraseTextFromCrop(cropped, i + 1);
        images.push(cleaned || cropped.toString('base64'));
      }
    } catch (e) {
      console.error(`[Storyboard] Vision: crop error for panel:`, e.message);
      images.push(null);
    }
  }

  return { images, panelCentroids };
}

/**
 * Erase caption text from a single cropped panel image.
 * Calls frame_detector.py in erase_text mode — it runs tesseract on the crop,
 * finds text in the bottom 40%, and paints it with the background color.
 * Returns base64 JPEG of cleaned image, or null if no text found / error.
 */
async function eraseTextFromCrop(croppedBuffer, panelNum) {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ocr-'));
  const cropPath = path.join(tmpDir, 'crop.jpg');

  try {
    await fs.writeFile(cropPath, croppedBuffer);

    const result = await new Promise((resolve) => {
      const script = path.join(__dirname, 'frame_detector.py');
      const args = [script, cropPath, 'erase_text'];

      const tryPython = (cmd) => {
        const proc = spawn(cmd, args);
        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', d => stdout += d);
        proc.stderr.on('data', d => stderr += d);

        proc.on('close', code => {
          if (stderr) console.log(`[Storyboard] OCR-crop panel ${panelNum}: ${stderr.trim()}`);
          if (code !== 0) {
            if (cmd === 'python3') { tryPython('python'); return; }
            resolve(null);
            return;
          }
          try {
            resolve(JSON.parse(stdout));
          } catch (e) {
            resolve(null);
          }
        });

        proc.on('error', () => {
          if (cmd === 'python3') { tryPython('python'); return; }
          resolve(null);
        });
      };

      tryPython('python3');
    });

    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});

    if (result?.cleaned) {
      return result.cleaned;
    }
    return null;

  } catch (e) {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
    return null;
  }
}

/**
 * Match Claude's frame structure to pdf_panels.py's images using imageCount.
 *
 * Claude says: [{imageCount:1}, {imageCount:1}, {imageCount:3}, {imageCount:1}]
 * pdf_panels.py found: [img0, img1, img2, img3, img4, img5]
 *
 * Result:
 * - Frame 0 → img0 (single)
 * - Frame 1 → img1 (single)
 * - Frame 2 → [img2, img3, img4] (3-image pan/triptych → subImages)
 * - Frame 3 → img5 (single)
 */
function matchClaudeFramesToPdfImages(claudeFrames, pdfImages) {
  const matched = [];
  let imgIndex = 0;

  for (const cf of claudeFrames) {
    const imageCount = cf.imageCount || 1;

    if (imgIndex >= pdfImages.length) {
      // No more images — push frame with no image
      matched.push({
        frameNumber: cf.frameNumber || `${matched.length + 1}`,
        description: cf.description || '',
        dialog: cf.dialog || '',
        panelX: cf.panelX ?? null,
        panelY: cf.panelY ?? null,
        image: null,
        subImages: null
      });
      continue;
    }

    if (imageCount === 1) {
      matched.push({
        frameNumber: cf.frameNumber || `${matched.length + 1}`,
        description: cf.description || '',
        dialog: cf.dialog || '',
        panelX: cf.panelX ?? null,
        panelY: cf.panelY ?? null,
        image: pdfImages[imgIndex].image,
        subImages: null
      });
      imgIndex++;
    } else {
      // Multi-image frame (pan/tilt/track sequence)
      const available = Math.min(imageCount, pdfImages.length - imgIndex);
      const subImages = [];
      for (let i = 0; i < available; i++) {
        subImages.push(pdfImages[imgIndex + i].image);
      }
      matched.push({
        frameNumber: cf.frameNumber || `${matched.length + 1}`,
        description: cf.description || '',
        dialog: cf.dialog || '',
        panelX: cf.panelX ?? null,
        panelY: cf.panelY ?? null,
        image: subImages[0],
        subImages: subImages
      });
      imgIndex += available;
    }
  }

  // If there are leftover images pdf_panels.py found that Claude didn't account for,
  // append them as extra single-image frames
  while (imgIndex < pdfImages.length) {
    matched.push({
      frameNumber: `${matched.length + 1}`,
      description: '',
      dialog: '',
      panelX: null,
      panelY: null,
      image: pdfImages[imgIndex].image,
      subImages: null
    });
    imgIndex++;
  }

  return matched;
}

function groupIntoShots(frames) {
  const shots = [];
  let currentShot = null;
  let shotNumber = 1;

  // Check if Pass 2 AI grouping produced results
  const hasAIGrouping = frames.some(f => f.shotGroup !== undefined);

  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    let startNewShot = false;

    if (i === 0) {
      startNewShot = true;
    } else if (f.hasVisibleNumber) {
      // PRIORITY: When frames have visible numbers, use them as ground truth.
      // Each numbered frame = its own shot.
      // Multi-image sequences (pans/tilts) are already grouped within a single frame
      // via subImages from Vision-first, so no shot-level grouping needed.
      startNewShot = true;
    } else if (hasAIGrouping && f.shotGroup !== undefined && frames[i-1].shotGroup !== undefined) {
      // Fallback: AI visual grouping for boards without visible frame numbers
      startNewShot = f.shotGroup !== frames[i-1].shotGroup;
    } else {
      // No grouping data and no numbers — each frame is its own shot
      startNewShot = true;
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

    // For multi-image frames (diptych/triptych/etc.), push ALL sub-images
    // and repeat the frame number for each so frames[] and images[] stay aligned.
    // The client uses shot.frames[i] as the badge for shot.images[i].
    if (f.subImages && f.subImages.length > 1) {
      for (const subImg of f.subImages) {
        if (subImg) {
          currentShot.frames.push(f.frameNumber);
          currentShot.images.push(subImg);
        }
      }
    } else {
      currentShot.frames.push(f.frameNumber);
      if (f.image) currentShot.images.push(f.image);
    }
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
// Opus option — simple prompt, small thumbnails, ~$0.44/spot
async function analyzeGroupingsOpus(frames) {
  const client = getAnthropicClient();
  if (!client) return frames;

  const thumbs = [];
  for (let i = 0; i < frames.length; i++) {
    if (thumbs.length >= 40) break;
    const f = frames[i];
    if (f.image) {
      const thumb = await sharp(Buffer.from(f.image, 'base64'))
        .resize(500, 500, { fit: 'inside', withoutEnlargement: true })
        .jpeg({ quality: 80 })
        .toBuffer();
      thumbs.push({
        img: { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: thumb.toString('base64') } },
        frameIdx: i,
        desc: f.description || ''
      });
    }
  }

  if (thumbs.length < 2) return frames;
  console.log(`[Storyboard] Pass 2 (Sonnet): ${thumbs.length} frames`);

  const content = [];
  for (let i = 0; i < thumbs.length; i++) {
    const t = thumbs[i];
    content.push({ type: 'text', text: `Frame ${i + 1}${t.desc ? ` — "${t.desc.substring(0, 100)}"` : ''}:` });
    content.push(t.img);
  }

  const pairList = [];
  for (let i = 0; i < thumbs.length - 1; i++) {
    pairList.push(`${i + 1}→${i + 2}: SAME or CUT?`);
  }

  content.push({ type: 'text', text: `You are analyzing ${thumbs.length} storyboard frames from a TV commercial.

For each consecutive pair, decide SAME or CUT.

SAME shot means the camera is in the same setup:
- Same framing, same subject, same angle — action just progresses
- Arrows drawn on frames (any color) indicate camera MOVEMENT (push in, pull out, pan, tilt, dolly, zoom). Arrows = continuous shot, NOT a cut.

CUT means the camera moved to a new setup:
- Different framing (wide shot vs close-up)
- Different subject
- The BACKGROUND/OBJECTS visible behind the subject change dramatically — you see a completely different side of the environment. This means the camera moved to the opposite side.
- Different location

When in doubt, lean toward SAME. Storyboard frames from the same shot often look slightly different due to the artist's drawing — focus on whether the CAMERA SETUP changed, not small drawing variations.

${pairList.join('\n')}

Answer ONLY with the format: 1→2: SAME, 2→3: CUT, etc.` });

  try {
    const response = await apiCallWithRetry(() => client.messages.create({
      model: 'claude-sonnet-4-5-20250929',
      max_tokens: 2048,
      messages: [{ role: 'user', content }]
    }));
    return applyPass2Decisions(frames, thumbs, response.content[0]?.text || '');
  } catch (err) {
    console.error('[Storyboard] Pass 2 (Sonnet) error:', err.message);
    return frames;
  }
}

// Sonnet option — full descriptions, high-res images, ~$0.10/spot
async function analyzeGroupings(frames) {
  const client = getAnthropicClient();
  if (!client) return frames;

  // Build high-res images (1200px) for better visual detail
  const thumbs = [];
  for (let i = 0; i < frames.length; i++) {
    if (thumbs.length >= 40) break;
    const f = frames[i];
    if (f.image) {
      const thumb = await sharp(Buffer.from(f.image, 'base64'))
        .resize(1200, 1200, { fit: 'inside', withoutEnlargement: true })
        .jpeg({ quality: 85 })
        .toBuffer();
      thumbs.push({
        img: { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: thumb.toString('base64') } },
        frameIdx: i,
        desc: f.description || ''
      });
    }
  }

  if (thumbs.length < 2) return frames;
  console.log(`[Storyboard] Pass 2 (dual-pass ensemble): ${thumbs.length} frames`);

  // Build shared image content block (reused by both passes)
  const imageContent = [];
  for (let i = 0; i < thumbs.length; i++) {
    const t = thumbs[i];
    const descText = t.desc ? `\nDescription: "${t.desc}"` : '';
    imageContent.push({ type: 'text', text: `Frame ${i + 1}:${descText}` });
    imageContent.push(t.img);
  }

  // --- Pass A: Camera position labels (spatial reasoning) ---
  const positionContent = [...imageContent];
  positionContent.push({ type: 'text', text: `You are analyzing ${thumbs.length} storyboard frames from a TV commercial.

For each frame, picture the MAIN SUBJECT as a 3D object (a person, a car, a table — whatever dominates the frame). Ask yourself: which SIDE or SURFACE of that subject am I looking at? Assign a camera setup label (A, B, C, etc.) based on this.

Rules:
- Same side/surface of the same subject at similar scale = SAME letter, even if the action progresses.
- Arrows drawn on frames indicate continuous camera movement = same letter.
- If you see a person from BEHIND (back of head, over their shoulder, looking past them) in one frame, and then you see that person's FACE in another frame — those MUST be different letters. The camera physically moved to the other side of them.
- Same principle for objects: front of a car vs rear, one side of a desk vs the other = different letter.
- Dramatic changes in shot size = different letter. An extreme close-up (just hands, just eyes) requires the camera to be MUCH closer than a wide shot — different setup.
- A wider shot that reveals MORE of the same scene (same subjects still visible, just more context around them) can be the same letter IF the camera is looking from the same direction.
- DO NOT think in terms of "sides of the room." Think in terms of "which surface of the subject am I seeing."

When in doubt, use the SAME letter — but if the subject's orientation toward camera clearly changed (back→front, left side→right side), use a DIFFERENT letter.

For each frame, answer with the letter AND a brief reason explaining where the camera is:
Frame 1: A — camera is behind the desk, looking at the group
Frame 2: A — same position, action progresses
Frame 3: B — camera moved to close-up position near the subject
...etc.` });

  // --- Pass B: Pairwise SAME/CUT (direct comparison) ---
  const pairContent = [...imageContent];
  const pairList = [];
  for (let i = 0; i < thumbs.length - 1; i++) {
    pairList.push(`${i + 1}→${i + 2}: SAME or CUT?`);
  }
  pairContent.push({ type: 'text', text: `You are analyzing ${thumbs.length} storyboard frames from a TV commercial.

For each consecutive pair, look ONLY at the two images and decide SAME or CUT. Ignore the text descriptions — judge purely by what the camera sees.

SAME = the two frames show essentially the same view. Same shot size, same angle, same subject — only the action or pose changed slightly. Arrows drawn between frames indicate continuous camera movement = SAME.

CUT = the view changed significantly. ANY of these = CUT:
- Shot size changed dramatically (close-up vs wide, detail vs full figure)
- DIFFERENT SIDE of a person or object: Mentally picture each subject as a 3D object. If you see the BACK or SIDE of someone (over-the-shoulder, behind them, looking past them) in one frame and their FACE or FRONT in the next — the camera moved ~180°, that is ALWAYS a cut. Same for objects: front of a car vs rear, one side of a desk vs the other.
- Different subject or different part of a subject (boots vs crown vs face)
- Different background or environment visible

${pairList.join('\n')}

Answer ONLY with the format: 1→2: SAME, 2→3: CUT, etc.` });

  // --- Pass C: Subject/content identification ---
  const subjectContent = [...imageContent];
  subjectContent.push({ type: 'text', text: `You are analyzing ${thumbs.length} storyboard frames from a TV commercial.

For each frame, identify WHAT the camera is focused on — the primary subject that fills the frame. Then assign a subject label (A, B, C, etc.) where frames showing the same subject at roughly the same scale get the same letter.

Different label when:
- The subject changes (e.g. boots → full figure → crown → face)
- The scale changes dramatically (extreme close-up of a detail vs wide shot of a person)
- We see a completely different part of the scene

Same label when:
- Same subject at similar scale, just a different moment or pose

For each frame, answer with the letter AND what the subject is:
Frame 1: A — extreme close-up of floor/rug pattern
Frame 2: B — close-up of boots walking
Frame 3: B — same boots, slightly wider
...etc.` });

  // Run all three passes in PARALLEL
  try {
    const [positionRes, pairRes, subjectRes] = await Promise.all([
      apiCallWithRetry(() => client.messages.create({
        model: 'claude-sonnet-4-5-20250929',
        max_tokens: 2048,
        messages: [{ role: 'user', content: positionContent }]
      })),
      apiCallWithRetry(() => client.messages.create({
        model: 'claude-sonnet-4-5-20250929',
        max_tokens: 2048,
        messages: [{ role: 'user', content: pairContent }]
      })),
      apiCallWithRetry(() => client.messages.create({
        model: 'claude-sonnet-4-5-20250929',
        max_tokens: 2048,
        messages: [{ role: 'user', content: subjectContent }]
      }))
    ]);

    const positionText = positionRes.content[0]?.text || '';
    const pairText = pairRes.content[0]?.text || '';
    const subjectText = subjectRes.content[0]?.text || '';

    // Flexible label parser — extracts labels in order of appearance
    // Handles: **Frame 1: A**, **Frame 1.1: A**, Frame 2 (1.2): B, etc.
    function parseLabels(text, count) {
      // Find all "Frame <anything>: <LETTER>" patterns in order
      const allMatches = [...text.matchAll(/Frame\s+[\d.]+(?:\s*\([^)]*\))?[^:\n]*?:\s*\**\s*([A-Z])\b/gi)];
      const labels = [];
      for (let i = 0; i < count; i++) {
        labels.push(allMatches[i] ? allMatches[i][1].toUpperCase() : null);
      }
      return labels;
    }

    // Parse Pass A: position labels → cuts
    console.log(`[Storyboard] Pass A (positions) raw:`, positionText);
    const positionCuts = new Set();
    const posLabels = parseLabels(positionText, thumbs.length);
    for (let i = 1; i < posLabels.length; i++) {
      if (posLabels[i] && posLabels[i - 1] && posLabels[i] !== posLabels[i - 1]) {
        positionCuts.add(i);
      }
    }
    console.log(`[Storyboard] Pass A labels: ${posLabels.join(', ')} → cuts at: [${[...positionCuts].join(', ')}]`);

    // Parse Pass B: SAME/CUT decisions → cuts
    console.log(`[Storyboard] Pass B (pairwise) raw:`, pairText);
    const pairCuts = new Set();
    // Extract all SAME/CUT pairs in order of appearance
    const allPairs = [...pairText.matchAll(/[\d.]+\s*[→>:\-]+\s*[\d.]+\s*[:.]\s*(SAME|CUT)/gi)];
    for (let i = 0; i < Math.min(allPairs.length, thumbs.length - 1); i++) {
      if (allPairs[i][1].toUpperCase() === 'CUT') {
        pairCuts.add(i + 1);
      }
    }
    console.log(`[Storyboard] Pass B cuts at: [${[...pairCuts].join(', ')}]`);

    // Parse Pass C: subject labels → cuts
    console.log(`[Storyboard] Pass C (subjects) raw:`, subjectText);
    const subjectCuts = new Set();
    const subLabels = parseLabels(subjectText, thumbs.length);
    for (let i = 1; i < subLabels.length; i++) {
      if (subLabels[i] && subLabels[i - 1] && subLabels[i] !== subLabels[i - 1]) {
        subjectCuts.add(i);
      }
    }
    console.log(`[Storyboard] Pass C labels: ${subLabels.join(', ')} → cuts at: [${[...subjectCuts].join(', ')}]`);

    // Merge: 2-out-of-3 vote — cut if at least 2 passes agree
    const allBoundaries = new Set([...positionCuts, ...pairCuts, ...subjectCuts]);
    const mergedCuts = new Set();
    for (const b of allBoundaries) {
      const votes = (positionCuts.has(b) ? 1 : 0) + (pairCuts.has(b) ? 1 : 0) + (subjectCuts.has(b) ? 1 : 0);
      if (votes >= 2) mergedCuts.add(b);
    }
    console.log(`[Storyboard] Votes: ${[...allBoundaries].sort((a,b) => a-b).map(b => `${b}=${(positionCuts.has(b)?'A':'')+(pairCuts.has(b)?'B':'')+(subjectCuts.has(b)?'C':'')}`).join(', ')}`);
    console.log(`[Storyboard] Merged cuts (2/3 vote): [${[...mergedCuts].sort((a,b) => a-b).join(', ')}]`);

    // Apply merged cuts to frames
    let shotGroup = 0;
    frames[thumbs[0].frameIdx].shotGroup = shotGroup;
    for (let i = 1; i < thumbs.length; i++) {
      if (mergedCuts.has(i)) shotGroup++;
      frames[thumbs[i].frameIdx].shotGroup = shotGroup;
    }

    console.log(`[Storyboard] Pass 2: ${shotGroup + 1} shots (3-pass ensemble)`);

    // Sanity check: if 0 cuts from 5+ frames, discard
    if (shotGroup === 0 && thumbs.length >= 5) {
      console.log(`[Storyboard] Pass 2: 0 cuts from ${thumbs.length} frames — discarding`);
      frames.forEach(f => { delete f.shotGroup; });
    }
  } catch (err) {
    console.error('[Storyboard] Pass 2 error:', err.message);
  }

  return frames;
}

// Parse camera position labels (A, B, C...) and derive shotGroups
function applyPositionLabels(frames, thumbs, text) {
  console.log(`[Storyboard] Pass 2 raw:`, text);

  // Parse "Frame N: X" lines — extract labels in document order
  const allMatches = [...text.matchAll(/Frame\s+[\d.]+(?:\s*\([^)]*\))?[^:\n]*?:\s*\**\s*([A-Z])\b/gi)];
  const labels = [];
  for (let i = 0; i < thumbs.length; i++) {
    labels.push(allMatches[i] ? allMatches[i][1].toUpperCase() : null);
  }

  console.log(`[Storyboard] Pass 2 labels:`, labels.join(', '));

  // Convert to shotGroups: new group whenever label changes
  let shotGroup = 0;
  frames[thumbs[0].frameIdx].shotGroup = shotGroup;

  for (let i = 1; i < thumbs.length; i++) {
    if (labels[i] && labels[i - 1] && labels[i] !== labels[i - 1]) {
      shotGroup++;
    }
    frames[thumbs[i].frameIdx].shotGroup = shotGroup;
  }

  console.log(`[Storyboard] Pass 2: ${shotGroup + 1} shots from position labels`);

  // Sanity check: if 0 cuts from 5+ frames, discard
  if (shotGroup === 0 && thumbs.length >= 5) {
    console.log(`[Storyboard] Pass 2: 0 cuts from ${thumbs.length} frames — discarding`);
    frames.forEach(f => { delete f.shotGroup; });
  }

  return frames;
}

// Shared: parse SAME/CUT decisions and apply shotGroup to frames
function applyPass2Decisions(frames, thumbs, text) {
  console.log(`[Storyboard] Pass 2 raw:`, text);

  let shotGroup = 0;
  frames[thumbs[0].frameIdx].shotGroup = shotGroup;
  const decisions = [];

  // Extract all SAME/CUT decisions in order of appearance
  const allPairs = [...text.matchAll(/[\d.]+\s*[→>:\-]+\s*[\d.]+\s*[:.]\s*(SAME|CUT)/gi)];
  for (let i = 0; i < thumbs.length - 1; i++) {
    const decision = allPairs[i] ? allPairs[i][1].toUpperCase() : 'SAME';
    decisions.push(decision);
    if (decision === 'CUT') shotGroup++;
    frames[thumbs[i + 1].frameIdx].shotGroup = shotGroup;
  }

  console.log(`[Storyboard] Pass 2: ${decisions.join(', ')} → ${shotGroup + 1} shots`);

  // Sanity check: if 0 cuts from 5+ frames, discard
  if (shotGroup === 0 && thumbs.length >= 5) {
    console.log(`[Storyboard] Pass 2: 0 cuts from ${thumbs.length} frames — discarding`);
    frames.forEach(f => { delete f.shotGroup; });
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

    // === PRE-PHASE: Convert presentations (PPTX/KEY) to PDF ===
    // LibreOffice converts to PDF, then we run the normal PDF pipeline.
    // This avoids a separate code path for presentations.
    let fileBuffer = req.file.buffer;
    let effectiveMimetype = req.file.mimetype;

    const isPresentation = isPresentationFile(req.file.mimetype, req.file.originalname);
    const isPptx = isPresentation && (req.file.originalname || '').toLowerCase().endsWith('.pptx');
    let pptxPanelResults = null;  // Direct PPTX image extraction (better quality than PDF re-encoding)

    if (isPresentation) {
      // For .pptx files: extract images directly using pptx_panels.py (original quality)
      if (isPptx) {
        try {
          pptxPanelResults = await extractPanelsFromPptxStructure(req.file.buffer, tempDir);
        } catch (e) {
          console.error('[Storyboard] pptx_panels.py error:', e.message);
        }
      }

      // Convert to PDF via LibreOffice (for page rendering + Claude Vision)
      try {
        console.log('[Storyboard] Converting presentation to PDF via LibreOffice...');
        const conversion = await convertPresentationToPdf(req.file.buffer, req.file.originalname, tempDir);
        fileBuffer = conversion.pdfBuffer;
        effectiveMimetype = 'application/pdf';
        console.log('[Storyboard] Presentation converted to PDF successfully');
      } catch (e) {
        console.error('[Storyboard] Presentation conversion failed:', e.message);
        return res.status(400).json({ error: `Failed to convert presentation: ${e.message}` });
      }
    }

    // === PHASE 0: Try PDF structure extraction first (pdf_panels.py) ===
    // This reads image placement coordinates directly from the PDF,
    // giving pixel-perfect crops with proper caption association.
    let pdfStructureResults = null;
    let pdfStructurePagesHandled = new Set(); // Pages fully handled by pdf_panels.py

    if (effectiveMimetype === 'application/pdf') {
      try {
        pdfStructureResults = await extractPanelsFromPdfStructure(fileBuffer, tempDir);

        if (pdfStructureResults) {
          for (const { pageNum, data } of pdfStructureResults.pages) {
            if (data && data.count > 0) {
              // VISION-FIRST: pdf_panels.py extracts images, Claude determines structure.
              // Mark every page where pdf_panels.py found real panel images as "handled".
              //
              // EXCEPTION: Scanned PDFs have ONE huge image per page (the scan itself).
              // If a page has only 1 image covering >70% of the page, it's probably a
              // scan — skip it so Vision/OpenCV can detect individual panels within.
              const images = data.images || data.panels || [];
              if (images.length === 1) {
                const img = images[0];
                const pageW = data.pageWidth || 0;
                const pageH = data.pageHeight || 0;
                if (pageW > 0 && pageH > 0) {
                  const imgArea = img.width * img.height;
                  const pageArea = pageW * pageH;
                  const coverage = imgArea / pageArea;
                  if (coverage > 0.7) {
                    console.log(`[Storyboard] Page ${pageNum}: pdf_panels.py found 1 image covering ${Math.round(coverage * 100)}% of page — likely a scan, falling back to Vision/OpenCV`);
                    continue;
                  }
                }
              }
              pdfStructurePagesHandled.add(pageNum);
              console.log(`[Storyboard] Page ${pageNum}: pdf_panels.py extracted ${images.length} images — Claude will determine structure`);
            }
          }
        }
      } catch (e) {
        console.error('[Storyboard] pdf_panels.py error:', e.message);
      }

      // Free memory: release base64 image data from pages NOT handled by pdf_panels.py.
      // Scanned PDFs have full-page images at zoom=3 that can be 1-3MB each in base64.
      // Holding these while Puppeteer launches causes OOM on memory-constrained containers.
      if (pdfStructureResults) {
        if (pdfStructurePagesHandled.size === 0) {
          // No pages handled — release everything
          console.log('[Storyboard] No pages handled by pdf_panels.py — releasing scan image memory');
          pdfStructureResults = null;
        } else {
          // Release image data from unhandled pages only
          for (const { pageNum, data } of pdfStructureResults.pages) {
            if (!pdfStructurePagesHandled.has(pageNum) && data && data.images) {
              data.images = []; // Free base64 strings
              data.count = 0;
            }
          }
        }
      }
    }

    // For PPTX files: prefer pptx_panels.py results over pdf_panels.py
    // pptx_panels.py extracts original-quality images directly from the PPTX,
    // while pdf_panels.py would extract from the LibreOffice-converted PDF (lossy).
    if (pptxPanelResults && pptxPanelResults.pages.some(p => p.data.count > 0)) {
      console.log('[Storyboard] Using pptx_panels.py images (original quality) over pdf_panels.py');
      pdfStructureResults = pptxPanelResults;
      pdfStructurePagesHandled = new Set();
      for (const { pageNum, data } of pptxPanelResults.pages) {
        if (data && data.count > 0) {
          pdfStructurePagesHandled.add(pageNum);
        }
      }
    }

    // === PHASE 1: Render ALL pages ===
    // Always render pages so Claude Vision can extract text from them.
    // pdf_panels.py handles images; Claude handles ALL text extraction.
    let pageImages = [];
    if (effectiveMimetype === 'application/pdf') {
      pageImages = await pdfToImages(fileBuffer, imageDir);
    } else {
      const imgPath = path.join(imageDir, 'page-1.png');
      await sharp(req.file.buffer).png().toFile(imgPath);
      pageImages = [imgPath];
    }

    // Determine total page count
    const totalPageCount = pdfStructureResults ? pdfStructureResults.totalPages : pageImages.length;

    console.log(`[Storyboard] ${totalPageCount} page(s) - ${pdfStructurePagesHandled.size} handled by pdf_panels.py, ${totalPageCount - pdfStructurePagesHandled.size} need Vision/OpenCV`);

    // === PHASE 2: Process pages that still need Vision/OpenCV pipeline ===
    const BATCH_SIZE = 1;
    const CONCURRENCY = 2;
    const allPageResults = [];

    // First, build results for pdf_panels.py pages (images only — Claude handles all text/structure)
    if (pdfStructureResults) {
      for (const { pageNum, data } of pdfStructureResults.pages) {
        if (!data || data.count < 1) continue;
        if (!pdfStructurePagesHandled.has(pageNum)) continue;

        // pdf_panels.py now returns simple {images: [{x, y, width, height, image}]}
        const pdfImages = data.images || [];

        allPageResults.push({
          detected: {
            count: pdfImages.length,
            bordered: false,
            mode: 'pdf_structure',
            pdfImages,  // Raw image list from pdf_panels.py (reading order)
            images: pdfImages.map(img => img.image),  // Base64 strings for downstream compatibility
            panelCentroids: pdfImages.map(img => {
              const pageW = data.pageWidth || 1;
              const pageH = data.pageHeight || 1;
              return {
                cx: (img.x + img.width / 2) / pageW,
                cy: (img.y + img.height / 2) / pageH
              };
            })
          },
          textData: {
            // Placeholder — Claude will fill this in via Vision-first matching
            frames: pdfImages.map(() => ({
              frameNumber: '', description: '', dialog: '',
              panelX: null, panelY: null
            })),
            boardType: 'photo',
            spotName: null,
            hasVisibleNumbers: false
          },
          pageNum,
          fromPdfStructure: true
        });
      }
    }

    // Now process remaining pages via the Vision/OpenCV pipeline
    const fallbackPages = [];
    for (let i = 0; i < pageImages.length; i++) {
      const pageNum = i + 1;
      if (!pdfStructurePagesHandled.has(pageNum)) {
        fallbackPages.push({ path: pageImages[i], pageNum });
      }
    }

    // VISION-FIRST: Send ALL pdf_panels.py pages to Claude for structure + text.
    // Claude determines which images are separate frames vs multi-image sequences.
    // Pages where Claude sees significantly more frames than pdf_panels.py found
    // will be re-routed to the Vision/OpenCV fallback pipeline.
    const reroutedPages = [];
    const pdfStructurePagesForClaude = [];
    for (let i = 0; i < pageImages.length; i++) {
      const pageNum = i + 1;
      if (pdfStructurePagesHandled.has(pageNum)) {
        pdfStructurePagesForClaude.push({ path: pageImages[i], pageNum });
      }
    }

    if (pdfStructurePagesForClaude.length > 0) {
      const textBatches = [];
      for (let i = 0; i < pdfStructurePagesForClaude.length; i += BATCH_SIZE) {
        textBatches.push(pdfStructurePagesForClaude.slice(i, i + BATCH_SIZE));
      }

      for (let i = 0; i < textBatches.length; i += CONCURRENCY) {
        const batchGroup = textBatches.slice(i, i + CONCURRENCY);
        await Promise.all(batchGroup.map(async (batch) => {
          const textResults = await extractTextBatched(batch, { includeStructure: true });

          for (const textData of textResults) {
            const existing = allPageResults.find(r => r.pageNum === textData.pageNum && r.fromPdfStructure);
            if (!existing) continue;

            // Update metadata from Claude
            existing.textData.spotName = textData.spotName || null;
            existing.textData.boardType = textData.boardType || 'photo';
            existing.textData.hasVisibleNumbers = textData.hasVisibleNumbers || false;
            existing.textData.gridLayout = textData.gridLayout;

            const pdfImages = existing.detected.pdfImages || [];
            const claudeFrames = textData.frames || [];
            const claudeFrameCount = claudeFrames.length;

            if (claudeFrameCount === 0) {
              console.log(`[Storyboard] Page ${textData.pageNum}: Claude returned no frames — keeping ${pdfImages.length} images as individual frames`);
              continue;
            }

            // Calculate total imageCount from Claude's structure
            const totalImageCount = claudeFrames.reduce((sum, f) => sum + (f.imageCount || 1), 0);
            const pdfImageCount = pdfImages.length;

            if (totalImageCount === pdfImageCount) {
              // Perfect match — use Vision-first matching
              const matched = matchClaudeFramesToPdfImages(claudeFrames, pdfImages);

              // Update detected images to reflect Claude's grouping
              existing.detected.count = matched.length;
              existing.detected.images = matched.map(f => {
                if (f.subImages) return f.subImages;  // Array for multi-image frames
                return f.image;  // String for single-image frames
              });

              // Update text data
              existing.textData.frames = matched.map(f => ({
                frameNumber: f.frameNumber,
                description: f.description,
                dialog: f.dialog,
                panelX: f.panelX,
                panelY: f.panelY
              }));

              const multiImageFrames = matched.filter(f => f.subImages);
              console.log(`[Storyboard] Page ${textData.pageNum}: Vision-first matched ${claudeFrameCount} frames to ${pdfImageCount} images` +
                (multiImageFrames.length > 0 ? ` (${multiImageFrames.length} multi-image sequences)` : ''));
            } else if (totalImageCount > pdfImageCount * 1.5) {
              // Claude sees significantly more frames than pdf_panels.py found.
              // The PDF structure doesn't match the visual layout (e.g. hand-drawn
              // frames composited into a few large image blocks). Re-route this
              // page to Vision/OpenCV which can detect panels visually.
              console.log(`[Storyboard] Page ${textData.pageNum}: Claude sees ${totalImageCount} images but pdf_panels.py found ${pdfImageCount} — re-routing to Vision/OpenCV`);

              // Remove this page from allPageResults so it goes through fallback
              const existingIdx = allPageResults.indexOf(existing);
              if (existingIdx >= 0) allPageResults.splice(existingIdx, 1);

              // Add to fallback pages (rendered image already available from Phase 1)
              const pageImg = pageImages[textData.pageNum - 1];
              if (pageImg) {
                reroutedPages.push({ path: pageImg, pageNum: textData.pageNum });
              }
            } else {
              // Minor count mismatch — fall back to index matching
              console.log(`[Storyboard] Page ${textData.pageNum}: imageCount mismatch (Claude total: ${totalImageCount}, pdf images: ${pdfImageCount}) — falling back to index matching`);
              const maxLen = Math.max(claudeFrameCount, pdfImageCount);
              const mergedFrames = [];
              const mergedImages = [];
              for (let fi = 0; fi < maxLen; fi++) {
                const cf = (fi < claudeFrameCount) ? claudeFrames[fi] : {};
                mergedFrames.push({
                  frameNumber: cf.frameNumber || '',
                  description: cf.description || '',
                  dialog: cf.dialog || '',
                  panelX: cf.panelX ?? null,
                  panelY: cf.panelY ?? null
                });
                mergedImages.push(fi < pdfImageCount ? pdfImages[fi].image : null);
              }
              existing.detected.count = mergedImages.length;
              existing.detected.images = mergedImages;
              existing.textData.frames = mergedFrames;
            }
          }
        }));
      }
    }

    // Add re-routed pages (where pdf_panels.py undercounted) to fallback pipeline
    if (reroutedPages.length > 0) {
      console.log(`[Storyboard] Re-routing ${reroutedPages.length} page(s) to Vision/OpenCV (pdf_panels.py undercounted)`);
      fallbackPages.push(...reroutedPages);
      // Sort by page number to maintain order
      fallbackPages.sort((a, b) => a.pageNum - b.pageNum);
    }

    // Process fallback pages (not handled by pdf_panels.py)
    const batches = [];
    for (let i = 0; i < fallbackPages.length; i += BATCH_SIZE) {
      batches.push(fallbackPages.slice(i, i + BATCH_SIZE));
    }

    for (let i = 0; i < batches.length; i += CONCURRENCY) {
      const batchGroup = batches.slice(i, i + CONCURRENCY);

      const batchResults = await Promise.all(batchGroup.map(async (batch) => {
        // Run batched text extraction (one API call for multiple pages)
        const textResults = await extractTextBatched(batch);

        // Detection: hand-drawn boards → OpenCV grid (pixel-perfect), photo boards → Vision
        return Promise.all(batch.map(async ({ path: imgPath, pageNum }) => {
          const textData = textResults.find(r => r.pageNum === pageNum) || { frames: [] };
          const textFrameCount = textData.frames?.length || 0;
          const boardType = textData.boardType || 'photo';

          let detected = { count: 0, bordered: false, mode: 'none', images: [], panelCentroids: [] };
          
          if (textFrameCount >= 1) {
            // Hand-drawn boards: try OpenCV grid first (pixel-perfect for ink/pencil borders)
            if (boardType === 'hand_drawn') {
              try {
                const cvResult = await detectRectangles(imgPath);
                if (cvResult.mode === 'grid') {
                  const expectedMin = Math.max(1, Math.ceil(textFrameCount * 0.9));
                  if (cvResult.count >= expectedMin && cvResult.images && cvResult.images.length >= 1) {
                    // Compute normalized centroids from grid rectangles
                    const imgMeta = await sharp(await fs.readFile(imgPath)).metadata();
                    const gridCentroids = (cvResult.rectangles || []).map(r => ({
                      cx: (r.x + r.width / 2) / imgMeta.width,
                      cy: (r.y + r.height / 2) / imgMeta.height
                    }));
                    detected = { ...cvResult, panelCentroids: gridCentroids };
                    console.log(`[Storyboard] Page ${pageNum}: hand-drawn → OpenCV grid found ${cvResult.count} panels (expected ~${textFrameCount})`);
                  } else {
                    console.log(`[Storyboard] Page ${pageNum}: hand-drawn → OpenCV grid found only ${cvResult.count}/${textFrameCount} panels — falling back to Vision`);
                  }
                } else {
                  console.log(`[Storyboard] Page ${pageNum}: hand-drawn but no grid detected — falling back to Vision`);
                }
              } catch (e) {
                console.error(`[Storyboard] Page ${pageNum}: OpenCV error:`, e.message);
              }
            } else {
              console.log(`[Storyboard] Page ${pageNum}: photo board → using Vision`);
            }
            
            // Vision fallback (or primary for photo boards)
            if (detected.count < 1) {
              try {
                const imageBuffer = await fs.readFile(imgPath);
                const visionResult = await detectPanelsWithVision(imageBuffer, textFrameCount, boardType);
                const visionImages = visionResult.images || [];
                const visionCentroids = visionResult.panelCentroids || [];
                if (visionImages.length >= 1) {
                  detected = { count: visionImages.length, bordered: false, mode: 'vision', images: visionImages, panelCentroids: visionCentroids };
                  console.log(`[Storyboard] Page ${pageNum}: Vision found ${visionImages.length} panels`);
                }
              } catch (e) {
                console.error(`[Storyboard] Page ${pageNum}: Vision error:`, e.message);
              }
            }
          }

          return { detected, textData, pageNum };
        }));
      }));
      
      allPageResults.push(...batchResults.flat());
    }
    
    // Post-processing: run text erasure on grid-path images
    // (Vision and Mask paths already do this during detection)
    // Skip for hand-drawn boards — Tesseract misreads ink strokes as text
    for (const result of allPageResults) {
      const boardType = result.textData?.boardType || 'photo';
      if (result.detected.mode === 'grid' && result.detected.images && boardType !== 'hand_drawn') {
        const cleaned = await Promise.all(result.detected.images.map(async (img, i) => {
          if (!img) return null;
          try {
            const buf = Buffer.from(img, 'base64');
            const cleanedB64 = await eraseTextFromCrop(buf, i + 1);
            return cleanedB64 || img;
          } catch (e) {
            return img;
          }
        }));
        result.detected.images = cleaned;
      }
    }
    
    // Sort by page number to maintain order
    allPageResults.sort((a, b) => a.pageNum - b.pageNum);

    // === SPOT RECONCILIATION ===
    // Each page was analyzed independently, so Claude may have returned scene headers
    // ("EXT. ALLEYWAY - NIGHT") or continuation text as spotNames when they're not
    // actually new commercials/spots. Do a lightweight text-only pass to reconcile.
    const pageSpotNames = allPageResults.map(r => ({
      pageNum: r.pageNum,
      spotName: r.textData?.spotName || null,
      frameCount: r.textData?.frames?.length || 0,
      boardType: r.textData?.boardType || 'photo'
    }));
    const rawSpotNames = pageSpotNames.filter(p => p.spotName);

    if (rawSpotNames.length > 1) {
      // Multiple pages returned spotNames — reconcile them
      try {
        const reconciledSpots = await reconcileSpotNames(pageSpotNames, totalPageCount);
        if (reconciledSpots) {
          // Apply reconciled spot names back to allPageResults
          for (const r of allPageResults) {
            const reconciled = reconciledSpots.find(s => s.pageNum === r.pageNum);
            if (reconciled && reconciled.spotName) {
              r.textData.spotName = reconciled.spotName;
            } else if (reconciled && reconciled.spotName === null) {
              r.textData.spotName = null; // Explicitly cleared — not a real spot boundary
            }
          }
        }
      } catch (e) {
        console.error('[Storyboard] Spot reconciliation error:', e.message);
        // Continue with original per-page spotNames
      }
    } else if (rawSpotNames.length === 1) {
      // Only one page has a spotName — apply it to all pages (single-spot document)
      const singleSpot = rawSpotNames[0].spotName;
      console.log(`[Storyboard] Single spot detected: "${singleSpot}"`);
    }

    const allFrames = [];
    let currentSpot = null;
    
    for (const { detected, textData, pageNum } of allPageResults) {
      if (textData.spotName) currentSpot = textData.spotName;
      
      const textFrames = textData.frames || [];
      const images = detected.images || [];
      const panelCentroids = detected.panelCentroids || [];
      const hasVisibleNumbers = textData.hasVisibleNumbers === true;
      
      // === PROXIMITY MATCHING ===
      // Match text frames to detected panels by spatial position instead of blind index
      // ONLY use proximity when counts differ — when equal, both sides are in reading order
      const hasPanelPositions = panelCentroids.length === images.length && panelCentroids.length > 0;
      const hasTextPositions = textFrames.some(tf => tf.panelX != null && tf.panelY != null);
      const countsDiffer = textFrames.length !== images.length;
      
      if (countsDiffer && hasPanelPositions && hasTextPositions && textFrames.length > 0 && images.length > 0) {
        // Proximity matching: pair each panel with its nearest text frame
        const usedTextIndices = new Set();
        const panelToText = new Array(images.length).fill(null);
        
        // For each panel, find closest unmatched text frame
        // Sort panels by Y then X (reading order) for deterministic matching
        const panelOrder = panelCentroids.map((c, i) => ({ idx: i, cx: c.cx, cy: c.cy }));
        panelOrder.sort((a, b) => {
          const rowDiff = Math.abs(a.cy - b.cy);
          if (rowDiff < 0.08) return a.cx - b.cx; // Same row: sort by X
          return a.cy - b.cy; // Different rows: sort by Y
        });
        
        for (const panel of panelOrder) {
          let bestDist = Infinity;
          let bestIdx = -1;
          
          for (let t = 0; t < textFrames.length; t++) {
            if (usedTextIndices.has(t)) continue;
            const tf = textFrames[t];
            if (tf.panelX == null || tf.panelY == null) continue;
            
            // Distance between panel centroid and text-reported panel position
            const dx = panel.cx - (tf.panelX / 100);
            const dy = panel.cy - (tf.panelY / 100);
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < bestDist) {
              bestDist = dist;
              bestIdx = t;
            }
          }
          
          if (bestIdx >= 0 && bestDist < 0.4) { // Max 40% of page distance
            panelToText[panel.idx] = bestIdx;
            usedTextIndices.add(bestIdx);
          }
        }
        
        // Also handle text frames without panelX/panelY — match by remaining index order
        const unmatchedPanels = [];
        for (let i = 0; i < images.length; i++) {
          if (panelToText[i] === null) unmatchedPanels.push(i);
        }
        const unmatchedTexts = [];
        for (let t = 0; t < textFrames.length; t++) {
          if (!usedTextIndices.has(t)) unmatchedTexts.push(t);
        }
        // Match remaining by order
        for (let k = 0; k < Math.min(unmatchedPanels.length, unmatchedTexts.length); k++) {
          panelToText[unmatchedPanels[k]] = unmatchedTexts[k];
        }
        
        // Build frames from matched pairs
        for (let i = 0; i < images.length; i++) {
          const tIdx = panelToText[i];
          const tf = tIdx !== null ? textFrames[tIdx] : {};
          const imgData = images[i];
          // Handle triptych: imgData can be an array of sub-images or a single string
          const primaryImage = Array.isArray(imgData) ? imgData[0] : imgData;
          const subImages = Array.isArray(imgData) ? imgData : null;
          allFrames.push({
            frameNumber: tf.frameNumber || `${i + 1}`,
            hasVisibleNumber: hasVisibleNumbers,
            description: tf.description || '',
            dialog: tf.dialog || '',
            image: primaryImage,
            subImages: subImages,
            spotName: currentSpot,
            pageNum: pageNum
          });
        }
        
        const matched = panelToText.filter(t => t !== null).length;
        console.log(`[Storyboard] Page ${pageNum}: PROXIMITY matched ${matched}/${images.length} panels to ${textFrames.length} text frames`);
        
      } else {
        // Index-based matching: counts match (reading order) or positions unavailable
        if (textFrames.length !== images.length && textFrames.length > 0 && images.length > 0) {
          console.log(`[Storyboard] Page ${pageNum}: COUNT MISMATCH — text ${textFrames.length}, panels ${images.length} (index fallback)`);
        } else if (textFrames.length === images.length && textFrames.length > 0) {
          console.log(`[Storyboard] Page ${pageNum}: INDEX matched ${images.length} panels to ${textFrames.length} text frames (counts equal)`);
        }
        
        const maxLen = Math.max(textFrames.length, images.length);
        for (let j = 0; j < maxLen; j++) {
          const tf = textFrames[j] || {};
          const imgData = images[j] || null;
          // Handle triptych: imgData can be an array of sub-images or a single string
          const primaryImage = Array.isArray(imgData) ? imgData[0] : imgData;
          const subImages = Array.isArray(imgData) ? imgData : null;

          allFrames.push({
            frameNumber: tf.frameNumber || `${j + 1}`,
            hasVisibleNumber: hasVisibleNumbers,
            description: tf.description || '',
            dialog: tf.dialog || '',
            image: primaryImage,
            subImages: subImages,
            spotName: currentSpot,
            pageNum: pageNum
          });
        }
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
      // BUT skip frames without real visible numbers (like title pages) — they shouldn't
      // trigger renumbering of the entire set.
      const numberPageMap = {};
      let hasDuplicates = false;
      for (const f of frames) {
        if (!f.hasVisibleNumber) continue; // Skip title pages, unnumbered frames
        const key = f.frameNumber;
        if (numberPageMap[key] && numberPageMap[key] !== f.pageNum) {
          hasDuplicates = true;
          break;
        }
        numberPageMap[key] = f.pageNum;
      }
      
      // Renumber if needed — but ONLY when truly necessary
      // Preserve original numbering (including decimals like 1.1, 1.2, 3.1) when possible
      // Log frame numbers before any renumbering for debugging
      console.log(`[Storyboard] Spot "${name}": frame numbers before renumbering: [${frames.map(f => `${f.frameNumber}(p${f.pageNum})`).join(', ')}]`);

      if (hasDuplicates) {
        // Duplicate numbers across pages — BUT check if most numbered frames
        // have unique numbers (e.g. only a few collide due to per-page numbering).
        // If so, just renumber the conflicting ones instead of wiping everything.
        const numberedFrames = frames.filter(f => f.hasVisibleNumber);
        const uniqueNumbers = new Set(numberedFrames.map(f => f.frameNumber));

        if (uniqueNumbers.size > numberedFrames.length * 0.5) {
          // Most numbers are unique — keep them, just fix conflicts
          console.log(`[Storyboard] Spot "${name}": mostly unique numbers (${uniqueNumbers.size}/${numberedFrames.length}), fixing conflicts only`);
          const usedNumbers = new Set();
          let nextFallback = frames.length + 1;
          for (const f of frames) {
            if (!f.hasVisibleNumber || !f.frameNumber) {
              f.frameNumber = String(nextFallback++);
            } else if (usedNumbers.has(f.frameNumber)) {
              f.frameNumber = String(nextFallback++);
            } else {
              usedNumbers.add(f.frameNumber);
            }
          }
        } else {
          // Most numbers are duplicated (per-page sequential) — full renumber
          console.log(`[Storyboard] Spot "${name}": renumbering ${frames.length} frames (duplicate numbers across pages)`);
          frames.forEach((f, idx) => {
            f.frameNumber = String(idx + 1);
          });
        }
      } else if (anyWithoutNumbers) {
        // Some frames lack visible numbers — only renumber if MOST frames lack numbers
        const framesWithNumbers = frames.filter(f => f.hasVisibleNumber);
        const framesWithoutNumbers = frames.filter(f => !f.hasVisibleNumber);

        if (framesWithNumbers.length > framesWithoutNumbers.length) {
          // Most frames have numbers — just fill in gaps for unnumbered ones
          console.log(`[Storyboard] Spot "${name}": keeping visible numbers, filling ${framesWithoutNumbers.length} gaps`);
          let nextNum = 1;
          for (const f of frames) {
            if (!f.hasVisibleNumber || !f.frameNumber || f.frameNumber === '') {
              f.frameNumber = String(nextNum);
            }
            // Track the highest number we've seen for gap-filling
            const num = parseFloat(f.frameNumber);
            if (!isNaN(num)) nextNum = Math.floor(num) + 1;
          }
        } else {
          console.log(`[Storyboard] Spot "${name}": renumbering ${frames.length} frames (missing numbers: ${anyWithoutNumbers})`);
          frames.forEach((f, idx) => {
            f.frameNumber = String(idx + 1);
          });
        }
      }

      // Always run Pass 2 AI grouping for visual shot analysis
      await analyzeGroupings(frames);

      spots.push({
        name,
        shots: groupIntoShots(frames)
      });
    }
    
    res.json({ spots });
    
  } catch (error) {
    console.error('[Storyboard] Error:', error);
    const status = error?.status || 500;
    if (status === 529) {
      res.status(503).json({ error: 'The AI service is temporarily overloaded. Please try again in a moment.' });
    } else {
      res.status(500).json({ error: error.message || 'Unknown error processing storyboard' });
    }
  } finally {
    // Clean up temp directory
    try { await fs.rm(tempDir, { recursive: true, force: true }); } catch (e) {}
  }
});

/**
 * Reconcile spot names across pages.
 * Individual pages may return scene headers, location slugs, or continuation text
 * as spotNames. This text-only pass gives Claude document-level context to determine
 * which are actual spot/commercial boundaries.
 */
async function reconcileSpotNames(pageSpotNames, totalPageCount) {
  const client = getAnthropicClient();
  if (!client) return null;

  const pageList = pageSpotNames.map(p =>
    `Page ${p.pageNum}: "${p.spotName || '(no title)'}" — ${p.frameCount} frames, ${p.boardType}`
  ).join('\n');

  console.log(`[Storyboard] Reconciling spot names across ${totalPageCount} pages`);

  const response = await apiCallWithRetry(() => client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 2048,
    messages: [{ role: 'user', content: `You are analyzing a storyboard PDF with ${totalPageCount} pages. Each page was analyzed independently and returned a title/header. Your job is to determine which titles represent DIFFERENT COMMERCIALS/SPOTS vs. which are just scene headers, location slugs, or page continuations within the SAME commercial.

Here are the per-page titles extracted:
${pageList}

RULES:
- A "spot" or "commercial" is a distinct advertisement or creative piece. A multi-page storyboard PDF often contains boards for just ONE commercial, but sometimes contains 2-4 different commercials.
- Scene headers like "EXT. ALLEYWAY - NIGHT", "INT. KITCHEN - DAY", "SCENE 2", "SC. 3" are NOT spot names — they describe locations/scenes WITHIN a commercial.
- If all pages appear to be from the SAME commercial (just different scenes), return ONE spot name for all pages.
- If pages clearly belong to DIFFERENT commercials (different product names, campaign names, or completely unrelated titles), split them into separate spots.
- Page numbers, frame counts, and "continued" headers don't indicate new spots.
- If a title appears only on page 1 and subsequent pages have scene headers, all pages likely belong to the page-1 spot.
- When in doubt, keep pages in the SAME spot — false splits are worse than false merges.

Return JSON (no extra text):
[
  { "pageNum": 1, "spotName": "COMMERCIAL NAME" },
  { "pageNum": 2, "spotName": "COMMERCIAL NAME" },
  ...
]

Every page must appear exactly once. Pages belonging to the same commercial get the SAME spotName string.` }]
  }));

  const text = response.content[0].text;
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) || text.match(/\[[\s\S]*\]/);

  try {
    const parsed = JSON.parse(jsonMatch ? (jsonMatch[1] || jsonMatch[0]).trim() : text.trim());
    const results = Array.isArray(parsed) ? parsed : [parsed];
    const spotNames = [...new Set(results.map(r => r.spotName))];
    console.log(`[Storyboard] Spot reconciliation: ${spotNames.length} spot(s) — ${spotNames.map(s => `"${s}"`).join(', ')}`);
    return results;
  } catch (e) {
    console.error('[Storyboard] Spot reconciliation parse error:', e.message);
    return null;
  }
}

/**
 * Extract text from multiple pages in a single API call
 */
async function extractTextBatched(pages, { includeStructure = false } = {}) {
  const client = getAnthropicClient();
  if (!client) throw new Error('ANTHROPIC_API_KEY not set');
  
  // Prepare all images
  const imageContents = await Promise.all(pages.map(async ({ path: imagePath, pageNum }) => {
    const imageBuffer = await fs.readFile(imagePath);
    const resized = await sharp(imageBuffer)
      .resize(1400, 1400, { fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 85 })
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
  
  // Build the prompt — two variants:
  // 1. includeStructure=true (Vision-first path): Claude determines multi-image groupings via imageCount
  // 2. includeStructure=false (fallback path): simpler prompt, panel detection handled by OpenCV/Vision
  const structureStep = includeStructure ? `
STEP 5 - STRUCTURE DETECTION (CRITICAL):
For each DISTINCT FRAME, determine how many separate IMAGE PANELS compose it:
- imageCount: 1 = single image (normal frame — one image, one description)
- imageCount: 2 = diptych or two-image sequence (pan, tilt, track, push, pull)
- imageCount: 3 = triptych or three-image sequence
- imageCount: 4+ = longer sequence (rare)

Multi-image sequences are frames where 2+ adjacent images share ONE description/caption and show continuous camera motion (pan left/right, tilt up/down, track in/out, dolly, crane, etc.). The images are side-by-side in the same row.

If images have SEPARATE descriptions or captions, they are SEPARATE frames with imageCount: 1 each — even if they sit next to each other in the same row.

STEP 6 - EXTRACT:
For each frame, extract text and estimate the PRIMARY image's center position on the page as a percentage (0-100).` : `
STEP 5 - EXTRACT:
For each frame, identify which IMAGE/PANEL it belongs to and estimate that panel's center position on the page as a percentage (0-100).`;

  const structureJsonExample = includeStructure
    ? `{ "frameNumber": "1", "imageCount": 1, "description": "Action/direction text", "dialog": "CHARACTER: Spoken lines...", "panelX": 25, "panelY": 35 }`
    : `{ "frameNumber": "1", "description": "Action/direction text", "dialog": "CHARACTER: Spoken lines...", "panelX": 25, "panelY": 35 }`;

  const structureRules = includeStructure ? `
- imageCount: MUST be present for every frame. Default is 1 (single image).
- totalImageCount: sum of all imageCount values for the page. This MUST equal the total number of distinct image panels visible on the page.
- CRITICAL: The sum of all imageCount values across frames MUST equal the total number of IMAGE PANELS on the page. For example: a 2x3 grid with 6 separate shots = 6 frames each with imageCount:1. A page with 4 separate shots and 1 three-image pan = 4 frames with imageCount:1 + 1 frame with imageCount:3 = totalImageCount:7.` : `
- CRITICAL: Count ALL image panels on each page. The number of frames you return per page MUST match the total number of IMAGE PANELS in the grid layout (e.g. 2x3 = 6 frames, 4+5+4 = 13 frames). Do NOT skip any panels.`;

  const totalImageCountField = includeStructure ? `\n    "totalImageCount": 6,` : '';

  content.push({ type: 'text', text: `Extract storyboard data from each page above.

STEP 1 - SPOT/SCRIPT NAME (CRITICAL):
Look for a BOLD TITLE near the top of each page - this is the commercial/spot name.
These titles indicate DIFFERENT COMMERCIALS/SPOTS in the same PDF.
Always extract the title exactly as written - even if it's just a number.
This is NOT scene descriptions like "INT. KITCHEN" - those are scene headers within a commercial.

STEP 2 - BOARD TYPE:
Classify the drawing panels on each page:
- "hand_drawn": panels contain hand-drawn illustrations, sketches, pencil/ink drawings, or animatic-style artwork
- "photo": panels contain photographs, rendered images, composited images, or photographic reference

STEP 3 - GRID LAYOUT:
Identify the grid structure. Read frames LEFT-TO-RIGHT, then TOP-TO-BOTTOM.

STEP 4 - FRAME NUMBERS:
- If frames have visible numbers (1, 2, 1A, 1B, 1.1, 1.2, 3.1, etc.), use those EXACTLY as written
- Decimal numbers like 1.1, 1.2, 2.1, 3.1 are common in storyboards — preserve the decimal format exactly
- If NO visible numbers, number sequentially: 1, 2, 3, 4...
${structureStep}

Return a JSON array with one object per page:
[
  {
    "pageNum": 1,
    "spotName": "EXACT TITLE FROM PAGE" or null,
    "boardType": "hand_drawn" or "photo",
    "gridLayout": "2x3",
    "hasVisibleNumbers": true/false,${totalImageCountField}
    "frames": [
      ${structureJsonExample}
    ]
  },
  { "pageNum": 2, ... }
]

panelX/panelY = approximate CENTER of the IMAGE/PANEL (not the text) as percentage of page width/height (0=left/top, 100=right/bottom).

RULES:
- spotName: ALWAYS extract the bold title at top of page - this identifies which commercial/spot
- boardType: classify based on the DRAWING PANELS content, not the page layout
- description: action/camera direction text
- dialog: spoken lines with character prefix${structureRules}
- Include frames even if they have no text (use empty strings for description/dialog)
- Text may appear below, beside, or near its panel — pair each text block with its nearest image` });
  
  console.log(`[Storyboard] Batched API call for ${pages.length} pages`);
  
  const response = await apiCallWithRetry(() => client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 8192,
    messages: [{ role: 'user', content }]
  }));
  
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

// SHOTLIST / SCHEDULE EXTRACTION
app.post('/api/extract-shotlist', upload.single('pdf'), async (req, res) => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'shotlist-'));
  
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    if (!process.env.ANTHROPIC_API_KEY) return res.status(500).json({ error: 'ANTHROPIC_API_KEY not set' });
    
    console.log('[ShotList] Processing:', req.file.originalname);
    const startTime = Date.now();
    
    const imageDir = path.join(tempDir, 'images');
    await fs.mkdir(imageDir, { recursive: true });
    
    // Convert PDF pages to images
    let pageImages = [];
    if (req.file.mimetype === 'application/pdf') {
      pageImages = await pdfToImages(req.file.buffer, imageDir);
    } else {
      const imgPath = path.join(imageDir, 'page-1.png');
      await sharp(req.file.buffer).png().toFile(imgPath);
      pageImages = [imgPath];
    }
    
    console.log(`[ShotList] ${pageImages.length} page(s)`);
    
    const client = getAnthropicClient();
    if (!client) throw new Error('ANTHROPIC_API_KEY not set');
    
    let allShots = [];
    let detectedType = null;
    let detectedStart = '';
    
    for (let i = 0; i < pageImages.length; i++) {
      const imageBuffer = await fs.readFile(pageImages[i]);
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
            { type: 'text', text: `Extract a structured shot list or schedule from this production document.

First, determine the document type:
- "schedule" = has time blocks, call times, or a timeline of activities
- "shotlist" = numbered shots/scenes with descriptions

Return JSON (no extra text):
{
  "type": "schedule" or "shotlist",
  "scheduleStart": "6:00 AM",
  "entries": [
    {
      "title": "PHOTO SETUP",
      "duration": 270,
      "scene": "",
      "notes": ""
    }
  ]
}

FIELD RULES:

title: Read each activity block on the document and copy its EXACT TEXT verbatim, then strip ONLY times and durations from that text. Keep everything else exactly as written — same words, same capitalization, same punctuation.
  Example: document shows "LUNCH @12pm 1h" → title: "LUNCH" (stripped @12pm and 1h)
  Example: document shows "HERO SOLO 30m" → title: "HERO SOLO" (stripped 30m)
  Example: document shows "Reduced Crew 2nd Meal 6pm" → title: "Reduced Crew 2nd Meal" (stripped 6pm)
  Example: document shows "Location Prep 1.5h" → title: "Location Prep" (stripped 1.5h)
  Example: document shows "9pm Tail Lights" → title: "Tail Lights" (stripped 9pm)
  Example: document shows "BTS Camera Call 12pm" → title: "BTS Camera Call" (stripped 12pm)
  DO NOT paraphrase, summarize, or generate new descriptions. Copy the words from the document.

duration: Duration in MINUTES as a number. Convert from the document: 1.5h=90, 4.5hr=270, 30m=30, 2h=120. If no duration shown, estimate from the time block size on the timeline. Default 30 if unknown.

scene: Scene number or location if shown. Use "" if none.

notes: Crew notes, equipment, or other meaningful non-timing details. Use "" if none.

scheduleStart: The EARLIEST time any activity begins — NOT the crew call time. Pre-call activities like location prep, pre-light, or advance crew often start before call. Look at the timeline and use the first activity's start time. For example, if "Location Prep" starts at 6:00 AM but "CALL" is 7:30 AM, scheduleStart should be "6:00 AM".

CRITICAL:
- Include ALL entries — don't skip breaks, meals, setups, resets, or wrap
- The title must come from the actual text on the document, with only times/durations removed
- For multi-line activity blocks, combine the text lines into a single title
- Entries should be in chronological order` }
          ]
        }]
      });
      
      const text = response.content[0].text;
      const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
      try {
        const parsed = JSON.parse((jsonMatch ? jsonMatch[1] : text).trim());
        // Support both old "shots" and new "entries" format
        const pageEntries = parsed.entries || parsed.shots || [];
        // Re-number if multiple pages
        const offset = allShots.length;
        pageEntries.forEach((s, idx) => { s.number = String(offset + idx + 1); });
        allShots.push(...pageEntries);
        // Capture type and scheduleStart from first page
        if (i === 0) {
          detectedType = parsed.type || null;
          detectedStart = parsed.scheduleStart || '';
        }
      } catch (e) {
        console.error(`[ShotList] Page ${i + 1} parse error`);
      }
    }
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[ShotList] Extracted ${allShots.length} entries (${elapsed}s)`);
    
    res.json({ 
      type: detectedType,
      scheduleStart: detectedStart,
      entries: allShots,
      shots: allShots  // backward compat
    });
    
  } catch (error) {
    console.error('[ShotList] Error:', error);
    res.status(500).json({ error: error.message });
  } finally {
    try { await fs.rm(tempDir, { recursive: true, force: true }); } catch (e) {}
  }
});

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
    let documentUnionStatus = null;
    let unionEvidence = null;
    
    for (let i = 0; i < pageImages.length; i++) {
      const imagePath = pageImages[i];
      const pageNum = i + 1;
      
      console.log(`[Cast] Page ${pageNum}: analyzing...`);
      
      const imageBuffer = await fs.readFile(imagePath);
      
      // Step 1: Detect ALL faces on the page
      let faceBoxes = []; // { box, landmarks, cropImage, faceX, faceY }
      if (faceApiReady && faceapi) {
        try {
          faceBoxes = await detectAllFacesOnPage(imagePath);
          console.log(`[Cast] Page ${pageNum}: ${faceBoxes.length} faces detected`);
        } catch (err) {
          console.log(`[Cast] Page ${pageNum}: face detection error:`, err.message);
        }
      }
      
      // Fallback to rectangle detection if no faces found
      if (faceBoxes.length === 0) {
        console.log(`[Cast] Page ${pageNum}: no faces found, trying rectangle detection...`);
        const detected = await detectRectangles(imagePath);
        const rawImages = detected.images || [];
        
        for (let k = 0; k < rawImages.length; k++) {
          const img = rawImages[k];
          if (!img) continue;
          try {
            const cropResult = await cropToFace(img);
            if (cropResult) {
              faceBoxes.push({
                image: cropResult.image,
                faceX: cropResult.faceX,
                faceY: cropResult.faceY,
                pageX: detected.rectangles[k]?.x || 0,
                pageY: detected.rectangles[k]?.y || 0
              });
            }
          } catch (e) { /* skip */ }
        }
        console.log(`[Cast] Page ${pageNum}: ${faceBoxes.length} faces from rectangle fallback`);
      }
      
      // Step 2: Annotate the page image with numbered markers at each face
      let annotatedBuffer = imageBuffer;
      if (faceBoxes.length > 0) {
        try {
          annotatedBuffer = await annotatePageWithFaceNumbers(imageBuffer, faceBoxes);
          console.log(`[Cast] Page ${pageNum}: annotated image with ${faceBoxes.length} face markers`);
        } catch (err) {
          console.log(`[Cast] Page ${pageNum}: annotation error:`, err.message);
        }
      }
      
      // Step 3: Send annotated image to Claude - extract names AND match to face numbers
      const castData = await extractCastWithFaceMatching(annotatedBuffer, faceBoxes.length);
      const members = castData.members || [];
      
      // Track document-level union status
      if (castData.documentUnionStatus && !documentUnionStatus) {
        documentUnionStatus = castData.documentUnionStatus;
        unionEvidence = castData.unionEvidence || null;
        console.log(`[Cast] Document union status: ${documentUnionStatus} (${unionEvidence})`);
      }
      
      console.log(`[Cast] Page ${pageNum}: ${members.length} members matched by AI`);
      
      // Step 4: Build results using AI-matched face numbers
      for (let j = 0; j < members.length; j++) {
        const member = members[j];
        const faceIdx = (member.faceNumber || 0) - 1; // Convert 1-based to 0-based
        const headshot = faceBoxes[faceIdx] || null;
        const name = member.actorName || member.characterName || `Member ${j + 1}`;
        
        if (headshot) {
          console.log(`[Cast] ${name}: matched to face #${member.faceNumber}`);
        } else {
          console.log(`[Cast] ${name}: no headshot (face #${member.faceNumber} not found)`);
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
    
    res.json({ cast: allCast, documentUnionStatus, unionEvidence });
    
  } catch (error) {
    console.error('[Cast] Error:', error);
    res.status(500).json({ error: error.message });
  } finally {
    try { await fs.rm(tempDir, { recursive: true, force: true }); } catch (e) {}
  }
});

// Detect ALL faces on a full page image, crop each, return sorted in reading order
async function detectAllFacesOnPage(imagePath) {
  if (!faceApiReady || !faceapi) return [];
  
  const canvasModule = await import('canvas');
  const { createCanvas, loadImage } = canvasModule.default || canvasModule;
  
  const imageBuffer = await fs.readFile(imagePath);
  let img = await loadImage(imageBuffer);
  
  const imgWidth = img.width;
  const imgHeight = img.height;
  
  console.log(`[FaceAPI-All] Page image size: ${imgWidth}x${imgHeight}`);
  
  // Detect all faces
  const detections = await faceapi
    .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.3 }))
    .withFaceLandmarks(true);
  
  if (!detections || detections.length === 0) {
    console.log('[FaceAPI-All] No faces detected on page');
    return [];
  }
  
  console.log(`[FaceAPI-All] Found ${detections.length} faces`);
  
  // Sort faces in reading order: top-to-bottom rows, then left-to-right
  const rowThreshold = imgHeight * 0.08;
  const sortedDetections = [...detections].sort((a, b) => {
    const aY = a.detection.box.y + a.detection.box.height / 2;
    const bY = b.detection.box.y + b.detection.box.height / 2;
    const aX = a.detection.box.x;
    const bX = b.detection.box.x;
    if (Math.abs(aY - bY) < rowThreshold) return aX - bX;
    return aY - bY;
  });
  
  // Crop each face into a headshot
  const faces = [];
  for (const detection of sortedDetections) {
    const box = detection.box || detection.detection.box;
    if (!box) continue;
    
    // Get eye positions for better centering
    const landmarks = detection.landmarks;
    let centerX = box.x + box.width / 2;
    let centerY = box.y + box.height / 2;
    
    if (landmarks) {
      try {
        const leftEye = landmarks.getLeftEye();
        const rightEye = landmarks.getRightEye();
        centerX = (leftEye.reduce((s, p) => s + p.x, 0) / leftEye.length + 
                   rightEye.reduce((s, p) => s + p.x, 0) / rightEye.length) / 2;
        centerY = (leftEye.reduce((s, p) => s + p.y, 0) / leftEye.length + 
                   rightEye.reduce((s, p) => s + p.y, 0) / rightEye.length) / 2;
      } catch (e) { /* use box center */ }
    }
    
    // Tight crop: 1.5x face size (head + small margin)
    const faceSize = Math.max(box.width, box.height);
    const cropSize = Math.min(Math.round(faceSize * 1.5), Math.min(imgWidth, imgHeight));
    let cropX = Math.round(centerX - cropSize / 2);
    let cropY = Math.round(centerY - cropSize * 0.4); // Offset up for forehead
    
    // Clamp to image bounds
    cropX = Math.max(0, Math.min(cropX, imgWidth - cropSize));
    cropY = Math.max(0, Math.min(cropY, imgHeight - cropSize));
    const finalSize = Math.min(cropSize, imgWidth - cropX, imgHeight - cropY);
    
    if (finalSize < 50) continue;
    
    try {
      const croppedBuffer = await sharp(imageBuffer)
        .extract({ left: cropX, top: cropY, width: finalSize, height: finalSize })
        .resize(300, 300, { fit: 'cover' })
        .jpeg({ quality: 90 })
        .toBuffer();
      
      const faceXInCrop = Math.max(0, Math.min(1, (centerX - cropX) / finalSize));
      const faceYInCrop = Math.max(0, Math.min(1, (centerY - cropY) / finalSize));
      
      faces.push({
        image: croppedBuffer.toString('base64'),
        faceX: faceXInCrop,
        faceY: faceYInCrop,
        // Page-level coordinates for annotation
        pageCenterX: Math.round(centerX),
        pageCenterY: Math.round(centerY)
      });
      
      console.log(`[FaceAPI-All] Face ${faces.length}: center ${Math.round(centerX)},${Math.round(centerY)} box ${Math.round(box.width)}x${Math.round(box.height)}`);
    } catch (err) {
      console.log(`[FaceAPI-All] Crop error:`, err.message);
    }
  }
  
  return faces;
}

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
// Annotate page image with numbered circles at each detected face position
async function annotatePageWithFaceNumbers(imageBuffer, faceBoxes) {
  const canvasModule = await import('canvas');
  const { createCanvas, loadImage } = canvasModule.default || canvasModule;
  
  const img = await loadImage(imageBuffer);
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  
  // Draw original image
  ctx.drawImage(img, 0, 0);
  
  // Draw numbered circles at each face
  for (let i = 0; i < faceBoxes.length; i++) {
    const face = faceBoxes[i];
    const cx = face.pageCenterX || (face.pageX || 0);
    const cy = face.pageCenterY || (face.pageY || 0);
    
    // Red circle with white number
    const radius = Math.max(20, Math.min(40, img.width * 0.025));
    
    // Draw circle background
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 0, 0, 0.85)';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw number
    ctx.fillStyle = 'white';
    ctx.font = `bold ${Math.round(radius * 1.2)}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), cx, cy);
  }
  
  // Export as JPEG
  const buffer = canvas.toBuffer('image/jpeg', { quality: 0.85 });
  return buffer;
}

// Send annotated image to Claude - extracts names AND matches each to a face number
async function extractCastWithFaceMatching(imageBuffer, faceCount) {
  const client = getAnthropicClient();
  if (!client) throw new Error('ANTHROPIC_API_KEY not set');
  
  const resized = await sharp(imageBuffer)
    .resize(1600, 1600, { fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 90 })
    .toBuffer();
  
  const faceNote = faceCount > 0 
    ? `\nThe image has ${faceCount} RED NUMBERED CIRCLES overlaid on detected faces. Each circle contains a number (1, 2, 3...). Match each person's name to their face number by looking at which name is closest to which numbered circle.`
    : '';
  
  const response = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 4096,
    messages: [{
      role: 'user',
      content: [
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: resized.toString('base64') } },
        { type: 'text', text: `Extract cast/talent information from this cast sheet.${faceNote}

For each person, identify their name(s) and match to the correct face number.

CRITICAL RULES:
- Names can appear ABOVE, BELOW, or BESIDE their photo - determine by PROXIMITY, not by reading order
- If a person has TWO names shown (e.g. character name AND actor name), include both
- If only ONE name is shown per person, put it in actorName and leave characterName empty
- NEVER put "UNKNOWN" or "N/A" in any field - leave it as empty string "" instead
- faceNumber must match the red numbered circle on or near that person's photo
- If no numbered circles are visible, use left-to-right, top-to-bottom order (1, 2, 3...)

Return JSON:
{
  "members": [
    {
      "faceNumber": 1,
      "actorName": "JANE DOE",
      "characterName": "NINA",
      "role": "",
      "age": null,
      "dob": null,
      "isMinor": false,
      "hardIn": null,
      "hardOut": null,
      "unionStatus": null
    }
  ],
  "documentUnionStatus": null,
  "unionEvidence": null
}

FIELD GUIDE:
- faceNumber: The red circle number on this person's face (integer)
- actorName: Real person's name (the talent/actor). Use "" if not shown.
- characterName: Character/role name. Use "" if not shown.
- If only one name exists per person, it is most likely the ACTOR name - put it in actorName
- role: Additional role description if any
- age: Numeric age if shown. null if not shown.
- dob: Date of birth string if shown. null if not shown.
- isMinor: true if under 18 or listed as minor/child
- hardIn/hardOut: Call/wrap times if shown. null if not shown.
- unionStatus: "union" if SAG/AFTRA, "non-union" if specified, null if not shown
- documentUnionStatus: If the document header/title indicates a union status for the whole production, put it here
- unionEvidence: Quote the text that indicates union status` }
      ]
    }]
  });
  
  const text = response.content[0].text;
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  try {
    return JSON.parse((jsonMatch ? jsonMatch[1] : text).trim());
  } catch (e) {
    console.error('[Cast] Text parse error:', text.substring(0, 200));
    return { members: [] };
  }
}

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
      text: `You are analyzing storyboard frames from a commercial shoot. Your task is to identify which characters appear in each frame using BOTH the drawn image AND the accompanying text description.

CHARACTERS TO IDENTIFY: ${characters.join(', ')}
${headshotsAvailable > 0 ? '\nYou have reference photos of each character above. Match the storyboard drawings to these real faces - pay attention to gender, hair, and build.' : ''}

=== IMPORTANT: ROW NUMBERS ===

Each image above is labeled with its ROW NUMBER (e.g., "ROW 1", "ROW 2", etc.).
Some rows have MULTIPLE images - combine all characters from all images in that row.
Use the ROW NUMBERS from the labels - do NOT number images sequentially yourself.

There are exactly ${frames.length} rows to analyze. Your response must have exactly ${frames.length} assignments.

=== HOW TO IDENTIFY CHARACTERS ===

You have TWO sources of evidence. Use BOTH:

1. VISUAL: Look at the drawn figures. Match them to reference photos by gender, hair, build, and position.
2. TEXT: Read the description/dialogue text for each row. The text is written by the director and is AUTHORITATIVE about who is in the scene.

=== CRITICAL: TEXT OVERRIDES VISUAL AMBIGUITY ===

If the text description names a character (e.g. "KAREN enters the kitchen", "Rick visible in background"), that character IS in the frame — period. The text was written by the person who drew the board. They know who is in their own drawing.

When the text names a character:
- Look carefully for ANY figure that could be them, including background figures, partially visible people, silhouettes, or figures entering/exiting frame
- Count that figure in your bodyCount even if you initially missed it
- Tag that character

The ONLY reason to skip a text-named character is if the frame is genuinely empty or shows zero human figures at all.

=== ANALYSIS STEPS ===

For each ROW (using the row number from the label):
1. Read the text description FIRST — note which characters are named
2. Look at the image — find a figure for each named character
3. Also check for any additional figures not mentioned in text
4. Match all figures to characters using reference photos + text

RULES:
- Text-named characters should almost always be tagged — look harder for their figure
- ${headshotsAvailable > 0 ? 'Use reference photos to distinguish similar characters' : 'Build profiles from establishing shots'}
- CONTINUITY: Storyboard frames are SEQUENTIAL. A close-up (hands, object detail, tight shot) immediately following wider shots of a character is almost certainly still showing that same character. Tag them. For example: if frames 3-4 show Henry at a cabinet, and frame 5 is a close-up of hands holding a box, those are Henry's hands.
- Similarly, if a character is established in a scene and subsequent frames show the same location/action from different angles, assume they are still present unless the framing clearly excludes them

Respond with JSON:
{
  "assignments": [
    {"rowNum": "1", "bodyCount": 1, "characters": ["RICK"], "reasoning": "Text says RICK, 1 male figure in doorway matches"},
    {"rowNum": "2", "bodyCount": 3, "characters": ["RICK", "TANYA", "KAREN"], "reasoning": "Text names KAREN entering, 2 figures at table + 1 entering background"},
    ...
  ]
}

VALIDATION:
- Use the ROW NUMBERS from the image labels (ROW 1, ROW 2, etc.)
- You must have exactly ${frames.length} assignments
- Every character named in the text description should be tagged unless there is truly no figure present`
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
