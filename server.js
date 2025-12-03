// Version 1.1.0 - Added storyboard extraction

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

// Multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
  fileFilter: (req, file, cb) => {
    const allowed = ['application/pdf', 'image/jpeg', 'image/png', 'image/webp'];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Only PDF, JPEG, PNG, or WebP files allowed'), false);
    }
  }
});

// Initialize Anthropic client (lazy - only when needed)
let anthropic = null;
function getAnthropicClient() {
  if (!anthropic && process.env.ANTHROPIC_API_KEY) {
    anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY
    });
  }
  return anthropic;
}

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Health check endpoint
app.get('/', (req, res) => {
  res.json({ 
    status: 'ok', 
    message: 'Railway PDF Server running',
    version: '1.1.0',
    features: ['pdf-generation', 'storyboard-extraction']
  });
});

// PDF generation endpoint
app.post('/generate-pdf', async (req, res) => {
  let browser = null;
  
  try {
    const { html, orientation = 'landscape' } = req.body;
    
    if (!html) {
      return res.status(400).json({ error: 'HTML content is required' });
    }
    
    console.log('Launching browser...');
    console.log('Requested orientation:', orientation);
    
    browser = await puppeteer.launch({
      headless: 'new',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--disable-software-rasterizer',
        '--disable-extensions'
      ]
    });
    
    console.log('Browser launched, creating page...');
    const page = await browser.newPage();
    
    await page.setContent(html, {
      waitUntil: ['load', 'networkidle0']
    });
    
    console.log('Generating PDF with orientation:', orientation);
    
    const pdfBuffer = await page.pdf({
      format: 'Letter',
      landscape: orientation === 'landscape',
      printBackground: true,
      margin: {
        top: '0.4in',
        right: '0.4in',
        bottom: '0.4in',
        left: '0.4in'
      },
      preferCSSPageSize: true
    });
    
    console.log('PDF generated successfully');
    
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename="schedule.pdf"');
    res.send(pdfBuffer);
    
  } catch (error) {
    console.error('PDF generation error:', error);
    res.status(500).json({ 
      error: 'Failed to generate PDF',
      message: error.message
    });
  } finally {
    if (browser) {
      await browser.close();
      console.log('Browser closed');
    }
  }
});

// ============================================
// STORYBOARD EXTRACTION ENDPOINT
// ============================================

/**
 * Convert PDF to page images using Puppeteer
 */
async function pdfToPageImages(pdfBuffer, outputDir) {
  let browser = null;
  const images = [];
  
  try {
    browser = await puppeteer.launch({
      headless: 'new',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu'
      ]
    });
    
    const page = await browser.newPage();
    
    // Set viewport for reasonable resolution
    await page.setViewport({ width: 1200, height: 1600, deviceScaleFactor: 1.5 });
    
    // Create HTML page that renders PDF using PDF.js
    const base64Pdf = pdfBuffer.toString('base64');
    
    const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
      <style>
        body { margin: 0; background: white; }
        canvas { display: block; margin: 0 auto; }
      </style>
    </head>
    <body>
      <canvas id="canvas"></canvas>
      <script>
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        
        window.renderPage = async function(pageNum) {
          const pdfData = atob('${base64Pdf}');
          const pdfArray = new Uint8Array(pdfData.length);
          for (let i = 0; i < pdfData.length; i++) {
            pdfArray[i] = pdfData.charCodeAt(i);
          }
          
          const pdf = await pdfjsLib.getDocument({ data: pdfArray }).promise;
          window.totalPages = pdf.numPages;
          
          if (pageNum > pdf.numPages) return null;
          
          const pdfPage = await pdf.getPage(pageNum);
          const scale = 1.5; // Reduced from 2.0 for faster processing
          const viewport = pdfPage.getViewport({ scale });
          
          const canvas = document.getElementById('canvas');
          const context = canvas.getContext('2d');
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          
          await pdfPage.render({ canvasContext: context, viewport }).promise;
          return { width: canvas.width, height: canvas.height };
        };
        
        window.getPageCount = async function() {
          const pdfData = atob('${base64Pdf}');
          const pdfArray = new Uint8Array(pdfData.length);
          for (let i = 0; i < pdfData.length; i++) {
            pdfArray[i] = pdfData.charCodeAt(i);
          }
          const pdf = await pdfjsLib.getDocument({ data: pdfArray }).promise;
          return pdf.numPages;
        };
      </script>
    </body>
    </html>
    `;
    
    await page.setContent(html, { waitUntil: 'networkidle0' });
    
    // Wait for PDF.js to load
    await page.waitForFunction('typeof pdfjsLib !== "undefined"', { timeout: 10000 });
    
    // Get page count
    const pageCount = await page.evaluate('getPageCount()');
    console.log(`[Storyboard] PDF has ${pageCount} pages`);
    
    // Render each page
    for (let i = 1; i <= pageCount; i++) {
      console.log(`[Storyboard] Rendering page ${i}/${pageCount}`);
      
      const dimensions = await page.evaluate(`renderPage(${i})`);
      if (!dimensions) continue;
      
      // Wait for render
      await new Promise(r => setTimeout(r, 500));
      
      // Screenshot the canvas
      const canvas = await page.$('#canvas');
      const imagePath = path.join(outputDir, `page-${i}.jpg`);
      await canvas.screenshot({ path: imagePath, type: 'jpeg', quality: 85 });
      images.push(imagePath);
    }
    
    return images;
    
  } finally {
    if (browser) await browser.close();
  }
}

/**
 * Extract frame data from a page image using Claude
 */
async function extractPageData(imageBuffer) {
  const client = getAnthropicClient();
  if (!client) throw new Error('ANTHROPIC_API_KEY not configured');
  
  // Resize image for faster analysis (max 1500px on longest side)
  const resized = await sharp(imageBuffer)
    .resize(1500, 1500, { fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 80 })
    .toBuffer();
  
  const base64Image = resized.toString('base64');
  console.log(`[Storyboard] Sending image to Claude: ${Math.round(resized.length / 1024)}KB`);
  
  const response = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 4096,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: 'image/jpeg',
              data: base64Image
            }
          },
          {
            type: 'text',
            text: `Analyze this storyboard page and extract all frames.

For each frame, extract:
1. Frame number (e.g., "1", "1A", "2B") - look for labels like "Frame 1" or "FR 1A"
2. Spot/Scene name if visible at the top
3. Description - the action/direction text
4. Dialog - spoken text, VO, or copy with character names
5. Original - the complete text as it appears
6. Bounding box of the ILLUSTRATION/IMAGE area only (not the text) as percentages [x, y, width, height] from top-left corner

Return JSON:
{
  "spotName": "Spot name or null",
  "frames": [
    {
      "frameNumber": "1A",
      "description": "Martha talks to camera.",
      "dialog": "MARTHA: Hello...",
      "original": "Full original text...",
      "boundingBox": [5, 10, 28, 35]
    }
  ]
}

Be precise with bounding boxes - they should tightly fit each frame's illustration area.
If no frames found, return {"spotName": null, "frames": []}`
          }
        ]
      }
    ]
  });
  
  const text = response.content[0].text;
  let jsonStr = text;
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (jsonMatch) jsonStr = jsonMatch[1];
  
  try {
    return JSON.parse(jsonStr.trim());
  } catch (e) {
    console.error('Failed to parse response:', text);
    return { spotName: null, frames: [] };
  }
}

/**
 * Crop frame images based on bounding boxes
 */
async function cropFrameImages(pageImagePath, frames) {
  const imageBuffer = await fs.readFile(pageImagePath);
  const metadata = await sharp(imageBuffer).metadata();
  const { width, height } = metadata;
  
  const croppedImages = [];
  
  for (const frame of frames) {
    if (!frame.boundingBox || frame.boundingBox.length !== 4) {
      croppedImages.push(null);
      continue;
    }
    
    const [xPct, yPct, wPct, hPct] = frame.boundingBox;
    
    const left = Math.round((xPct / 100) * width);
    const top = Math.round((yPct / 100) * height);
    const cropWidth = Math.round((wPct / 100) * width);
    const cropHeight = Math.round((hPct / 100) * height);
    
    // Ensure bounds are valid
    const safeLeft = Math.max(0, Math.min(left, width - 1));
    const safeTop = Math.max(0, Math.min(top, height - 1));
    const safeWidth = Math.min(cropWidth, width - safeLeft);
    const safeHeight = Math.min(cropHeight, height - safeTop);
    
    if (safeWidth < 10 || safeHeight < 10) {
      croppedImages.push(null);
      continue;
    }
    
    try {
      const cropped = await sharp(imageBuffer)
        .extract({ left: safeLeft, top: safeTop, width: safeWidth, height: safeHeight })
        .resize(800, 800, { fit: 'inside', withoutEnlargement: true }) // Limit size
        .jpeg({ quality: 75 })
        .toBuffer();
      
      console.log(`[Storyboard] Cropped frame ${frame.frameNumber}: ${Math.round(cropped.length / 1024)}KB`);
      croppedImages.push(cropped.toString('base64'));
    } catch (e) {
      console.error('Failed to crop frame:', e);
      croppedImages.push(null);
    }
  }
  
  return croppedImages;
}

/**
 * Group frames into shots (1A, 1B, 1C -> Shot 1)
 */
function groupFramesIntoShots(allFrames) {
  const shotGroups = {};
  
  for (const frame of allFrames) {
    const match = frame.frameNumber?.match(/^(\d+)/);
    const shotNum = match ? match[1] : frame.frameNumber || 'unknown';
    
    if (!shotGroups[shotNum]) {
      shotGroups[shotNum] = {
        shotNumber: shotNum,
        frames: [],
        images: [],
        descriptions: [],
        dialogs: [],
        originals: []
      };
    }
    
    shotGroups[shotNum].frames.push(frame.frameNumber);
    if (frame.image) shotGroups[shotNum].images.push(frame.image);
    if (frame.description) shotGroups[shotNum].descriptions.push(frame.description);
    if (frame.dialog) shotGroups[shotNum].dialogs.push(frame.dialog);
    if (frame.original) shotGroups[shotNum].originals.push(frame.original);
  }
  
  return Object.values(shotGroups).map(group => ({
    shotNumber: group.shotNumber,
    frames: group.frames,
    images: group.images,
    description: group.descriptions.join('\n'),
    dialog: group.dialogs.join('\n'),
    original: group.originals.join('\n')
  }));
}

/**
 * Storyboard extraction endpoint
 */
app.post('/api/extract-storyboard', upload.single('pdf'), async (req, res) => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'storyboard-'));
  
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    if (!process.env.ANTHROPIC_API_KEY) {
      return res.status(500).json({ error: 'ANTHROPIC_API_KEY not configured' });
    }
    
    console.log('[Storyboard] Processing:', req.file.originalname, 'Type:', req.file.mimetype, 'Size:', req.file.size);
    
    let pageImages = [];
    const imageDir = path.join(tempDir, 'images');
    await fs.mkdir(imageDir, { recursive: true });
    
    // Handle based on file type
    if (req.file.mimetype === 'application/pdf') {
      // PDF: render pages to images
      pageImages = await pdfToPageImages(req.file.buffer, imageDir);
    } else {
      // Image: save directly
      const imagePath = path.join(imageDir, 'page-1.jpg');
      
      // Convert to JPEG and normalize
      await sharp(req.file.buffer)
        .jpeg({ quality: 90 })
        .toFile(imagePath);
      
      pageImages = [imagePath];
    }
    
    console.log(`[Storyboard] Processing ${pageImages.length} page(s)`);
    
    // Extract data from each page
    const allFrames = [];
    let currentSpotName = null;
    
    for (let i = 0; i < pageImages.length; i++) {
      console.log(`[Storyboard] Analyzing page ${i + 1}/${pageImages.length}`);
      
      const imageBuffer = await fs.readFile(pageImages[i]);
      const pageData = await extractPageData(imageBuffer);
      
      if (pageData.spotName) {
        currentSpotName = pageData.spotName;
      }
      
      // Crop frame images
      const croppedImages = await cropFrameImages(pageImages[i], pageData.frames);
      
      for (let j = 0; j < pageData.frames.length; j++) {
        allFrames.push({
          ...pageData.frames[j],
          image: croppedImages[j],
          spotName: currentSpotName
        });
      }
    }
    
    console.log(`[Storyboard] Extracted ${allFrames.length} frames`);
    
    // Group by spot name
    const spotGroups = {};
    for (const frame of allFrames) {
      const spotName = frame.spotName || 'Untitled Spot';
      if (!spotGroups[spotName]) {
        spotGroups[spotName] = [];
      }
      spotGroups[spotName].push(frame);
    }
    
    // Build final response
    const spots = Object.entries(spotGroups).map(([name, frames]) => ({
      name,
      shots: groupFramesIntoShots(frames)
    }));
    
    console.log(`[Storyboard] Returning ${spots.length} spots`);
    
    res.json({ spots });
    
  } catch (error) {
    console.error('[Storyboard] Error:', error);
    res.status(500).json({ error: error.message || 'Extraction failed' });
    
  } finally {
    try {
      await fs.rm(tempDir, { recursive: true, force: true });
    } catch (e) {
      console.error('Cleanup error:', e);
    }
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Railway PDF Server listening on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'production'}`);
  console.log(`Storyboard extraction: ${process.env.ANTHROPIC_API_KEY ? 'enabled' : 'disabled (no API key)'}`);
});
