// Version 1.1.0 - Added storyboard extraction

const express = require('express');
const puppeteer = require('puppeteer');
const cors = require('cors');
const multer = require('multer');
const Anthropic = require('@anthropic-ai/sdk');
const pdf = require('pdf-poppler');
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
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files allowed'), false);
    }
  }
});

// Initialize Anthropic client
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY
});

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
 * Convert PDF pages to images
 */
async function pdfToImages(pdfPath, outputDir) {
  const opts = {
    format: 'jpeg',
    out_dir: outputDir,
    out_prefix: 'page',
    page: null,
    scale: 2048
  };
  
  await pdf.convert(pdfPath, opts);
  
  const files = await fs.readdir(outputDir);
  const imageFiles = files
    .filter(f => f.startsWith('page') && f.endsWith('.jpg'))
    .sort((a, b) => {
      const numA = parseInt(a.match(/\d+/)?.[0] || '0');
      const numB = parseInt(b.match(/\d+/)?.[0] || '0');
      return numA - numB;
    });
  
  return imageFiles.map(f => path.join(outputDir, f));
}

/**
 * Convert image to base64
 */
async function imageToBase64(imagePath) {
  const buffer = await fs.readFile(imagePath);
  return buffer.toString('base64');
}

/**
 * Extract storyboard data from a single page using Claude
 */
async function extractPageData(imageBase64, pageNumber) {
  const response = await anthropic.messages.create({
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
              data: imageBase64
            }
          },
          {
            type: 'text',
            text: `Analyze this storyboard page and extract all frames/shots.

For each frame visible, extract:
1. Frame number (e.g., "1A", "2B", "4C") - look for labels like "FR 1A" or "FRAME 1A"
2. Spot/Scene name if visible (e.g., "Forever Young", "Morning Routine")
3. Description - the action/direction text describing what happens
4. Dialog - any spoken text, VO (voiceover), or copy. Include character names if shown
5. Original - the complete original text from under the frame (unparsed)
6. The approximate bounding box of the frame IMAGE as percentages [x, y, width, height] from top-left

Return JSON in this exact format:
{
  "spotName": "Name of spot/scene if visible, or null",
  "frames": [
    {
      "frameNumber": "1A",
      "description": "Martha talks to camera.",
      "dialog": "MARTHA: I've got big news...",
      "original": "Martha talks to camera. MARTHA: I've got big news...",
      "boundingBox": [10, 15, 30, 40]
    }
  ]
}

If no storyboard frames are found, return {"spotName": null, "frames": []}.
Be precise with frame numbers - they're usually in the format number+letter (1A, 2B, etc).`
          }
        ]
      }
    ]
  });
  
  const text = response.content[0].text;
  
  let jsonStr = text;
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (jsonMatch) {
    jsonStr = jsonMatch[1];
  }
  
  try {
    return JSON.parse(jsonStr.trim());
  } catch (e) {
    console.error('Failed to parse Claude response:', text);
    return { spotName: null, frames: [] };
  }
}

/**
 * Crop frame images from page based on bounding boxes
 */
async function cropFrameImages(pageImagePath, frames) {
  const image = sharp(pageImagePath);
  const metadata = await image.metadata();
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
    
    const safeLeft = Math.max(0, Math.min(left, width - 1));
    const safeTop = Math.max(0, Math.min(top, height - 1));
    const safeWidth = Math.min(cropWidth, width - safeLeft);
    const safeHeight = Math.min(cropHeight, height - safeTop);
    
    if (safeWidth < 10 || safeHeight < 10) {
      croppedImages.push(null);
      continue;
    }
    
    try {
      const cropped = await sharp(pageImagePath)
        .extract({ left: safeLeft, top: safeTop, width: safeWidth, height: safeHeight })
        .jpeg({ quality: 85 })
        .toBuffer();
      
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
    const shotNum = match ? match[1] : frame.frameNumber;
    
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
      return res.status(400).json({ error: 'No PDF file uploaded' });
    }
    
    if (!process.env.ANTHROPIC_API_KEY) {
      return res.status(500).json({ error: 'ANTHROPIC_API_KEY not configured' });
    }
    
    const pdfPath = path.join(tempDir, 'input.pdf');
    await fs.writeFile(pdfPath, req.file.buffer);
    
    console.log('[Storyboard] Processing PDF:', req.file.originalname);
    
    // Convert PDF to images
    const imageDir = path.join(tempDir, 'images');
    await fs.mkdir(imageDir, { recursive: true });
    
    const pageImages = await pdfToImages(pdfPath, imageDir);
    console.log(`[Storyboard] Converted ${pageImages.length} pages to images`);
    
    // Extract data from each page
    const allFrames = [];
    let currentSpotName = null;
    
    for (let i = 0; i < pageImages.length; i++) {
      console.log(`[Storyboard] Processing page ${i + 1}/${pageImages.length}`);
      
      const imageBase64 = await imageToBase64(pageImages[i]);
      const pageData = await extractPageData(imageBase64, i + 1);
      
      if (pageData.spotName) {
        currentSpotName = pageData.spotName;
      }
      
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
      console.error('Failed to cleanup temp dir:', e);
    }
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Railway PDF Server listening on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'production'}`);
  console.log(`Storyboard extraction: ${process.env.ANTHROPIC_API_KEY ? 'enabled' : 'disabled (no API key)'}`);
});
