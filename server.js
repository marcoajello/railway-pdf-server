// Version 2.6.0 - Added HTML generation for broadcast

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
  res.json({ status: 'ok', version: '2.6.0', features: ['pdf-generation', 'html-generation', 'storyboard-extraction'] });
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
// The input HTML from buildCompleteHTML already has embedded images and inline styles
// We just render it and return the cleaned version
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
    
    // Get the full rendered HTML with minimal changes
    const renderedHtml = await page.evaluate(() => {
      // Remove scripts only
      document.querySelectorAll('script').forEach(s => s.remove());
      
      // Remove interactive elements but preserve structure
      document.querySelectorAll('button, input[type="checkbox"]').forEach(el => el.remove());
      
      // Remove contenteditable and draggable attributes
      document.querySelectorAll('[contenteditable]').forEach(el => el.removeAttribute('contenteditable'));
      document.querySelectorAll('[draggable]').forEach(el => el.removeAttribute('draggable'));
      
      // Return the entire document HTML
      return document.documentElement.outerHTML;
    });
    
    // Return the full document
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
// STORYBOARD EXTRACTION
// ============================================

/**
 * Detect rectangles using Python/OpenCV
 */
async function detectRectangles(imagePath) {
  return new Promise((resolve) => {
    const script = path.join(__dirname, 'frame_detector.py');
    
    // Try python3 first, then python
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
 * Convert PDF to page images
 */
async function pdfToImages(pdfBuffer, outputDir) {
  let browser = null;
  const images = [];
  
  try {
    browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    });
    
    const page = await browser.newPage();
    page.setDefaultNavigationTimeout(120000); // 2 minutes
    page.setDefaultTimeout(120000);
    await page.setViewport({ width: 1400, height: 1800, deviceScaleFactor: 1.5 });
    
    const base64Pdf = pdfBuffer.toString('base64');
    const html = `<!DOCTYPE html><html><head>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
      <style>body{margin:0;background:white}canvas{display:block}</style>
    </head><body><canvas id="canvas"></canvas><script>
      pdfjsLib.GlobalWorkerOptions.workerSrc='https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
      window.renderPage=async function(n){
        const d=atob('${base64Pdf}'),a=new Uint8Array(d.length);
        for(let i=0;i<d.length;i++)a[i]=d.charCodeAt(i);
        const pdf=await pdfjsLib.getDocument({data:a}).promise;
        if(n>pdf.numPages)return null;
        const pg=await pdf.getPage(n),vp=pg.getViewport({scale:2}),c=document.getElementById('canvas');
        c.width=vp.width;c.height=vp.height;
        await pg.render({canvasContext:c.getContext('2d'),viewport:vp}).promise;
        return{w:c.width,h:c.height};
      };
      window.getPageCount=async function(){
        const d=atob('${base64Pdf}'),a=new Uint8Array(d.length);
        for(let i=0;i<d.length;i++)a[i]=d.charCodeAt(i);
        return(await pdfjsLib.getDocument({data:a}).promise).numPages;
      };
    </script></body></html>`;
    
    await page.setContent(html, { waitUntil: 'networkidle0', timeout: 120000 });
    await page.waitForFunction('typeof pdfjsLib!=="undefined"', { timeout: 60000 });
    
    const pageCount = await page.evaluate('getPageCount()');
    console.log(`[Storyboard] PDF: ${pageCount} pages`);
    
    for (let i = 1; i <= pageCount; i++) {
      await page.evaluate(`renderPage(${i})`);
      await new Promise(r => setTimeout(r, 400));
      const canvas = await page.$('#canvas');
      const tempPath = path.join(outputDir, `page-${i}-temp.png`);
      const imgPath = path.join(outputDir, `page-${i}.jpg`);
      
      // Take screenshot as PNG
      await canvas.screenshot({ path: tempPath, type: 'png' });
      
      // Compress to JPEG at 75% quality
      await sharp(tempPath)
        .jpeg({ quality: 75, progressive: true })
        .toFile(imgPath);
      
      // Delete temp PNG
      await fs.unlink(tempPath);
      
      images.push(imgPath);
    }
    
    return images;
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
        { type: 'text', text: `Extract text from this storyboard page.

Return JSON:
{
  "spotName": "Scene/spot name from header or null",
  "frames": [
    {
      "frameNumber": "1A",
      "description": "Action description text",
      "dialog": "CHAR: Spoken dialog..."
    }
  ]
}

RULES:
- List frames in reading order (left-to-right, top-to-bottom)
- frameNumber: the label like "1A.", "2B", "FR 3"
- description: the action/direction text
- dialog: spoken lines with character names
- Skip empty cells` }
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
    const match = f.frameNumber?.match(/^(\d+)/);
    const num = match ? match[1] : f.frameNumber || 'X';
    
    if (!groups[num]) {
      groups[num] = { shotNumber: num, frames: [], images: [], descriptions: [], dialogs: [] };
    }
    
    groups[num].frames.push(f.frameNumber);
    if (f.image) groups[num].images.push(f.image);
    if (f.description) groups[num].descriptions.push(f.description);
    if (f.dialog) groups[num].dialogs.push(f.dialog);
  }
  
  return Object.values(groups).map(g => ({
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
    
    // Process all pages in parallel
    const pageResults = await Promise.all(pageImages.map(async (imagePath, i) => {
      const pageNum = i + 1;
      console.log(`[Storyboard] Page ${pageNum}: starting...`);
      
      const imageBuffer = await fs.readFile(imagePath);
      
      // Run OpenCV and Claude in parallel for each page
      const [detected, textData] = await Promise.all([
        detectRectangles(imagePath),
        extractText(imageBuffer)
      ]);
      
      console.log(`[Storyboard] Page ${pageNum}: ${detected.count} rectangles, ${textData.frames?.length || 0} text frames`);
      
      return { detected, textData, pageNum };
    }));
    
    // Combine results in order
    const allFrames = [];
    let currentSpot = null;
    
    for (const { detected, textData } of pageResults) {
      if (textData.spotName) currentSpot = textData.spotName;
      
      const textFrames = textData.frames || [];
      const images = detected.images || [];
      
      const maxLen = Math.max(textFrames.length, images.length);
      for (let j = 0; j < maxLen; j++) {
        const tf = textFrames[j] || {};
        const img = images[j] || null;
        
        allFrames.push({
          frameNumber: tf.frameNumber || `${j + 1}`,
          description: tf.description || '',
          dialog: tf.dialog || '',
          image: img,
          spotName: currentSpot
        });
      }
    }
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[Storyboard] Total: ${allFrames.length} frames, ${allFrames.filter(f => f.image).length} with images (${elapsed}s)`);
    
    // Group by spot
    const spotGroups = {};
    for (const f of allFrames) {
      const spot = f.spotName || 'Untitled';
      if (!spotGroups[spot]) spotGroups[spot] = [];
      spotGroups[spot].push(f);
    }
    
    const spots = Object.entries(spotGroups).map(([name, frames]) => ({
      name,
      shots: groupIntoShots(frames)
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
