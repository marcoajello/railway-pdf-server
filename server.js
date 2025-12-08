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
app.use(express.json({ limit: '50mb' }));

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
    await page.setViewport({ width: 1200, height: 1600, deviceScaleFactor: 1 });
    
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

STEP 1 - SPOT NAME:
Look for the COMMERCIAL/PROJECT TITLE at the top of the page (e.g., "BRAND - PRODUCT :30").
This is NOT scene descriptions like "INT. KITCHEN" - those are scene headers within the commercial.

STEP 2 - GRID LAYOUT:
Identify the grid structure. Read frames LEFT-TO-RIGHT, then TOP-TO-BOTTOM.

STEP 3 - FRAME NUMBERS:
- If frames have visible numbers (1, 2, 1A, 1B, etc.), use those exactly
- If NO visible numbers, number sequentially: 1, 2, 3, 4...

STEP 4 - EXTRACT:
Return JSON:
{
  "spotName": "BRAND - PRODUCT :30" or null,
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
- spotName: Commercial title only, NOT scene headers
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
    
    console.log(`[Storyboard] ${pageImages.length} page(s) - starting parallel processing`);
    
    const pageResults = await Promise.all(pageImages.map(async (imagePath, i) => {
      const pageNum = i + 1;
      console.log(`[Storyboard] Page ${pageNum}: starting...`);
      
      const imageBuffer = await fs.readFile(imagePath);
      
      const [detected, textData] = await Promise.all([
        detectRectangles(imagePath),
        extractText(imageBuffer)
      ]);
      
      console.log(`[Storyboard] Page ${pageNum}: ${detected.count} rectangles, ${textData.frames?.length || 0} text frames`);
      
      return { detected, textData, pageNum };
    }));
    
    const allFrames = [];
    let currentSpot = null;
    
    for (const { detected, textData, pageNum } of pageResults) {
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
      const images = detected.images || [];
      
      for (let j = 0; j < members.length; j++) {
        const member = members[j];
        const img = images[j] || null;
        
        allCast.push({
          actorName: member.actorName || '',
          characterName: member.characterName || '',
          role: member.role || '',
          image: img
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

For each headshot/photo, identify:
1. ACTOR NAME - The real person's name
2. CHARACTER NAME - The role they play (if shown)
3. ROLE - Any role description (e.g., "BARTENDER", "GUY AT BAR")

Return JSON:
{
  "members": [
    {
      "actorName": "John Smith",
      "characterName": "Detective Jones",
      "role": "Lead"
    }
  ]
}

RULES:
- List members in reading order (left-to-right, top-to-bottom)
- actorName: The talent/actor's real name
- characterName: The character they play (may be same as role, or empty)
- role: Description like "BARTENDER", "MOM", etc.
- Skip any photos without names` }
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
// AUTO-TAG VISION ENDPOINT
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

app.listen(PORT, () => {
  console.log(`Server on port ${PORT}`);
  console.log(`Storyboard: ${process.env.ANTHROPIC_API_KEY ? 'enabled' : 'disabled'}`);
  console.log(`Hanging chad detection: ${process.env.ANTHROPIC_API_KEY ? 'enabled' : 'disabled'}`);
  console.log(`Cast import: ${process.env.ANTHROPIC_API_KEY ? 'enabled' : 'disabled'}`);
  console.log(`Auto-tag vision: ${process.env.ANTHROPIC_API_KEY ? 'enabled' : 'disabled'}`);
});
