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

STEP 1 - GRID LAYOUT:
First, identify the grid structure (e.g., 2x3 = 2 columns, 3 rows).
Read frames LEFT-TO-RIGHT, then TOP-TO-BOTTOM.

STEP 2 - FRAME NUMBERS:
- If frames have visible numbers (1, 2, 1A, 1B, FR3, etc.), use those exactly
- If NO numbers are visible, AUTO-NUMBER based on visual continuity:
  - Same shot continuing (camera move, same action evolving): 1A, 1B, 1C
  - New shot (cut to different angle/scene/character): increment to 2, 3, 4
  - Look for: same background, continuing motion, connected poses = SAME SHOT (A/B/C)
  - Look for: different location, new character, scene change = NEW SHOT (next number)

STEP 3 - EXTRACT:
Return JSON:
{
  "spotName": "Scene/spot title from header or null",
  "gridLayout": "2x3",
  "hasVisibleNumbers": true/false,
  "frames": [
    {
      "frameNumber": "1A",
      "description": "Action/direction text",
      "dialog": "CHARACTER: Spoken lines..."
    }
  ]
}

RULES:
- frameNumber: visible label OR auto-generated (1A, 1B, 2, 3A, 3B, etc.)
- description: action/camera direction text near the frame
- dialog: spoken lines with character name prefix
- Skip completely empty frames
- Preserve reading order strictly` }
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
  const groups = {};
  
  for (const f of frames) {
    // Extract base number: "1A" -> "1", "12B" -> "12", "FR 3" -> "3"
    const cleaned = (f.frameNumber || '').replace(/^(FR|FRAME|SHOT)[\s.]*/i, '');
    const match = cleaned.match(/^(\d+)/);
    const baseNum = match ? match[1] : null;
    
    // If no number found, use position-based key
    const groupKey = baseNum || `_pos_${Object.keys(groups).length + 1}`;
    
    if (!groups[groupKey]) {
      groups[groupKey] = { 
        shotNumber: baseNum || String(Object.keys(groups).length + 1), 
        frames: [], 
        images: [], 
        descriptions: [], 
        dialogs: [] 
      };
    }
    
    groups[groupKey].frames.push(f.frameNumber || groupKey);
    if (f.image) groups[groupKey].images.push(f.image);
    if (f.description) groups[groupKey].descriptions.push(f.description);
    if (f.dialog) groups[groupKey].dialogs.push(f.dialog);
  }
  
  // Sort by numeric value, then return
  const sorted = Object.entries(groups)
    .sort((a, b) => {
      const numA = parseInt(a[0]) || 999;
      const numB = parseInt(b[0]) || 999;
      return numA - numB;
    })
    .map(([_, g]) => g);
  
  return sorted.map(g => ({
    shotNumber: g.shotNumber,
    frames: g.frames,
    images: g.images,
    description: g.descriptions.join('\n'),
    dialog: g.dialogs.join('\n'),
    combined: [...g.descriptions, '', ...g.dialogs].filter(Boolean).join('\n')
  }));
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
    
    // For each spot, check if we need to renumber
    // If frames don't have visible numbers OR numbers repeat across pages, renumber sequentially
    const spots = Object.entries(spotGroups).map(([name, frames]) => {
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
      }
      
      return {
        name,
        shots: groupIntoShots(frames)
      };
    });
    
    res.json({ spots });
    
  } catch (error) {
    console.error('[Storyboard] Error:', error);
    res.status(500).json({ error: error.message });
  } finally {
    try { await fs.rm(tempDir, { recursive: true, force: true }); } catch (e) {}
  }
});

app.listen(PORT, () => {
  console.log(`Server on port ${PORT}`);
  console.log(`Storyboard: ${process.env.ANTHROPIC_API_KEY ? 'enabled' : 'disabled'}`);
  console.log(`Hanging chad detection: ${process.env.ANTHROPIC_API_KEY ? 'enabled' : 'disabled'}`);
});
