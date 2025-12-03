// Version 2.0.0 - Claude grid detection (no OpenCV)

const express = require('express');
const puppeteer = require('puppeteer');
const cors = require('cors');
const multer = require('multer');
const Anthropic = require('@anthropic-ai/sdk');
const sharp = require('sharp');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

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

// Health check
app.get('/', (req, res) => {
  res.json({ 
    status: 'ok', 
    version: '2.0.0',
    features: ['pdf-generation', 'storyboard-extraction']
  });
});

// PDF generation
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

// ============================================
// STORYBOARD EXTRACTION
// ============================================

async function pdfToPageImages(pdfBuffer, outputDir) {
  let browser = null;
  const images = [];
  
  try {
    browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    });
    
    const page = await browser.newPage();
    await page.setViewport({ width: 1200, height: 1600, deviceScaleFactor: 1.5 });
    
    const base64Pdf = pdfBuffer.toString('base64');
    const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
      <style>body { margin: 0; background: white; } canvas { display: block; }</style>
    </head>
    <body>
      <canvas id="canvas"></canvas>
      <script>
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        
        window.renderPage = async function(pageNum) {
          const pdfData = atob('${base64Pdf}');
          const pdfArray = new Uint8Array(pdfData.length);
          for (let i = 0; i < pdfData.length; i++) pdfArray[i] = pdfData.charCodeAt(i);
          
          const pdf = await pdfjsLib.getDocument({ data: pdfArray }).promise;
          if (pageNum > pdf.numPages) return null;
          
          const pdfPage = await pdf.getPage(pageNum);
          const viewport = pdfPage.getViewport({ scale: 1.5 });
          const canvas = document.getElementById('canvas');
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          
          await pdfPage.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;
          return { width: canvas.width, height: canvas.height };
        };
        
        window.getPageCount = async function() {
          const pdfData = atob('${base64Pdf}');
          const pdfArray = new Uint8Array(pdfData.length);
          for (let i = 0; i < pdfData.length; i++) pdfArray[i] = pdfData.charCodeAt(i);
          return (await pdfjsLib.getDocument({ data: pdfArray }).promise).numPages;
        };
      </script>
    </body>
    </html>`;
    
    await page.setContent(html, { waitUntil: 'networkidle0' });
    await page.waitForFunction('typeof pdfjsLib !== "undefined"', { timeout: 10000 });
    
    const pageCount = await page.evaluate('getPageCount()');
    console.log(`[Storyboard] PDF: ${pageCount} pages`);
    
    for (let i = 1; i <= pageCount; i++) {
      const dimensions = await page.evaluate(`renderPage(${i})`);
      if (!dimensions) continue;
      await new Promise(r => setTimeout(r, 300));
      
      const canvas = await page.$('#canvas');
      const imagePath = path.join(outputDir, `page-${i}.jpg`);
      await canvas.screenshot({ path: imagePath, type: 'jpeg', quality: 90 });
      images.push(imagePath);
    }
    
    return images;
  } finally {
    if (browser) await browser.close();
  }
}

/**
 * Ask Claude for grid layout + text extraction in one call
 */
async function analyzePageWithClaude(imageBuffer) {
  const client = getAnthropicClient();
  if (!client) throw new Error('ANTHROPIC_API_KEY not configured');
  
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
        {
          type: 'image',
          source: { type: 'base64', media_type: 'image/jpeg', data: resized.toString('base64') }
        },
        {
          type: 'text',
          text: `Analyze this storyboard page.

STEP 1 - COUNT THE GRID:
- How many COLUMNS of storyboard frames? (usually 2 or 3)
- How many ROWS of storyboard frames? (usually 2-4)
- Only count actual frame cells, not headers or titles

STEP 2 - EXTRACT TEXT for each cell (left-to-right, top-to-bottom):
- frameNumber: The label (e.g. "1A", "FR 2B", "3.")
- description: Action/direction text
- dialog: Spoken lines with character names (e.g. "MARTHA: Hello...")
- hasContent: true if cell has a frame, false if empty

Return JSON only:
{
  "spotName": "Spot/scene name from header, or null",
  "grid": { "cols": 2, "rows": 3 },
  "frames": [
    { "frameNumber": "1A", "description": "...", "dialog": "...", "hasContent": true }
  ]
}

RULES:
- frames.length MUST equal cols × rows
- Order: left-to-right, top-to-bottom (reading order)
- Empty cells: include with hasContent: false
- No grid (title page): return { "grid": null, "frames": [] }`
        }
      ]
    }]
  });
  
  const text = response.content[0].text;
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const jsonStr = jsonMatch ? jsonMatch[1] : text;
  
  try {
    return JSON.parse(jsonStr.trim());
  } catch (e) {
    console.error('[Storyboard] Parse error:', text.substring(0, 200));
    return { spotName: null, grid: null, frames: [] };
  }
}

/**
 * Slice image into grid cells based on detected layout
 */
async function sliceIntoGrid(imagePath, grid) {
  if (!grid?.cols || !grid?.rows) return [];
  
  const metadata = await sharp(imagePath).metadata();
  const { width, height } = metadata;
  
  // Margins - adjust if your storyboards have different headers
  const marginTop = Math.round(height * 0.06);
  const marginBottom = Math.round(height * 0.02);
  const marginSide = Math.round(width * 0.02);
  
  const gridWidth = width - (marginSide * 2);
  const gridHeight = height - marginTop - marginBottom;
  const cellWidth = Math.round(gridWidth / grid.cols);
  const cellHeight = Math.round(gridHeight / grid.rows);
  
  const cells = [];
  
  for (let row = 0; row < grid.rows; row++) {
    for (let col = 0; col < grid.cols; col++) {
      const left = marginSide + (col * cellWidth);
      const top = marginTop + (row * cellHeight);
      const inset = 4; // Small inset to avoid border lines
      
      try {
        const cropped = await sharp(imagePath)
          .extract({
            left: Math.max(0, left + inset),
            top: Math.max(0, top + inset),
            width: Math.min(cellWidth - inset * 2, width - left - inset),
            height: Math.min(cellHeight - inset * 2, height - top - inset)
          })
          .jpeg({ quality: 85 })
          .toBuffer();
        
        cells.push(cropped.toString('base64'));
      } catch (e) {
        console.error('[Storyboard] Crop error:', e.message);
        cells.push(null);
      }
    }
  }
  
  return cells;
}

function groupFramesIntoShots(allFrames) {
  const shotGroups = {};
  
  for (const frame of allFrames) {
    if (!frame.hasContent) continue;
    
    const match = frame.frameNumber?.match(/^(\d+)/);
    const shotNum = match ? match[1] : frame.frameNumber || 'X';
    
    if (!shotGroups[shotNum]) {
      shotGroups[shotNum] = { shotNumber: shotNum, frames: [], images: [], descriptions: [], dialogs: [] };
    }
    
    shotGroups[shotNum].frames.push(frame.frameNumber);
    if (frame.image) shotGroups[shotNum].images.push(frame.image);
    if (frame.description) shotGroups[shotNum].descriptions.push(frame.description);
    if (frame.dialog) shotGroups[shotNum].dialogs.push(frame.dialog);
  }
  
  return Object.values(shotGroups).map(g => ({
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
    
    console.log('[Storyboard] File:', req.file.originalname, req.file.mimetype);
    
    const imageDir = path.join(tempDir, 'images');
    await fs.mkdir(imageDir, { recursive: true });
    
    let pageImages = [];
    if (req.file.mimetype === 'application/pdf') {
      pageImages = await pdfToPageImages(req.file.buffer, imageDir);
    } else {
      const imagePath = path.join(imageDir, 'page-1.jpg');
      await sharp(req.file.buffer).jpeg({ quality: 90 }).toFile(imagePath);
      pageImages = [imagePath];
    }
    
    console.log(`[Storyboard] ${pageImages.length} page(s)`);
    
    const allFrames = [];
    let currentSpotName = null;
    
    for (let i = 0; i < pageImages.length; i++) {
      console.log(`[Storyboard] Page ${i + 1}/${pageImages.length}`);
      
      const imageBuffer = await fs.readFile(pageImages[i]);
      const pageData = await analyzePageWithClaude(imageBuffer);
      
      if (pageData.spotName) currentSpotName = pageData.spotName;
      
      if (!pageData.grid) {
        console.log(`[Storyboard] Page ${i + 1}: no grid`);
        continue;
      }
      
      console.log(`[Storyboard] Page ${i + 1}: ${pageData.grid.cols}x${pageData.grid.rows} = ${pageData.frames?.length || 0} frames`);
      
      const cellImages = await sliceIntoGrid(pageImages[i], pageData.grid);
      
      for (let j = 0; j < (pageData.frames || []).length; j++) {
        const frame = pageData.frames[j];
        if (!frame.hasContent) continue;
        
        allFrames.push({
          ...frame,
          image: cellImages[j] || null,
          spotName: currentSpotName
        });
      }
    }
    
    console.log(`[Storyboard] Total: ${allFrames.length} frames`);
    
    const spotGroups = {};
    for (const frame of allFrames) {
      const spot = frame.spotName || 'Untitled';
      if (!spotGroups[spot]) spotGroups[spot] = [];
      spotGroups[spot].push(frame);
    }
    
    const spots = Object.entries(spotGroups).map(([name, frames]) => ({
      name,
      shots: groupFramesIntoShots(frames)
    }));
    
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
});
