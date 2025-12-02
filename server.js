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
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files allowed'), false);
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
 * Convert PDF pages to images using Puppeteer
 */
async function pdfToImages(pdfBuffer, outputDir) {
  let browser = null;
  
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
    
    // Convert buffer to base64 data URL
    const base64Pdf = pdfBuffer.toString('base64');
    const dataUrl = `data:application/pdf;base64,${base64Pdf}`;
    
    // Navigate to PDF
    await page.goto(dataUrl, { waitUntil: 'networkidle0', timeout: 60000 });
    
    // Wait a moment for PDF to render
    await new Promise(r => setTimeout(r, 2000));
    
    // Get page count using PDF.js internals
    const pageCount = await page.evaluate(() => {
      // Try to get page count from PDF viewer
      const viewer = document.querySelector('#viewer');
      if (viewer) {
        const pages = viewer.querySelectorAll('.page');
        return pages.length || 1;
      }
      return 1;
    });
    
    console.log(`[Storyboard] PDF has approximately ${pageCount} pages`);
    
    // Take screenshot of entire page
    await page.setViewport({ width: 1600, height: 2000 });
    
    const images = [];
    
    // For single-page approach, just screenshot the whole thing
    const screenshotPath = path.join(outputDir, 'page-1.jpg');
    await page.screenshot({
      path: screenshotPath,
      fullPage: true,
      type: 'jpeg',
      quality: 90
    });
    images.push(screenshotPath);
    
    return images;
    
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

/**
 * Alternative: Use pdf-lib to extract pages, render with canvas
 * For now, send full PDF to Claude directly as it supports PDFs
 */
async function extractWithDirectPdf(pdfBuffer) {
  const client = getAnthropicClient();
  if (!client) {
    throw new Error('ANTHROPIC_API_KEY not configured');
  }
  
  const base64Pdf = pdfBuffer.toString('base64');
  
  const response = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 8192,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'document',
            source: {
              type: 'base64',
              media_type: 'application/pdf',
              data: base64Pdf
            }
          },
          {
            type: 'text',
            text: `Analyze this storyboard PDF and extract all frames/shots from all pages.

For each frame visible, extract:
1. Frame number (e.g., "1", "1A", "2B", "4C") - look for labels like "FR 1A", "FRAME 1A", or "Frame 1"
2. Spot/Scene name if visible at the top of the page
3. Description - the action/direction text describing what happens
4. Dialog - any spoken text, VO (voiceover), or copy. Include character names if shown
5. Original - the complete original text exactly as it appears under the frame

Return JSON in this exact format:
{
  "spots": [
    {
      "name": "Name of spot/scene, or 'Untitled' if not visible",
      "frames": [
        {
          "frameNumber": "1A",
          "description": "Martha talks to camera.",
          "dialog": "MARTHA: I've got big news...",
          "original": "Martha talks to camera. MARTHA: I've got big news..."
        }
      ]
    }
  ]
}

Extract ALL frames from ALL pages. Be thorough.
If a frame has no dialog, use empty string for dialog.
If a frame has no description, use empty string for description.`
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
    return { spots: [] };
  }
}

/**
 * Group frames into shots (1A, 1B, 1C -> Shot 1)
 */
function groupFramesIntoShots(frames) {
  const shotGroups = {};
  
  for (const frame of frames) {
    const match = frame.frameNumber?.match(/^(\d+)/);
    const shotNum = match ? match[1] : frame.frameNumber || 'unknown';
    
    if (!shotGroups[shotNum]) {
      shotGroups[shotNum] = {
        shotNumber: shotNum,
        frames: [],
        descriptions: [],
        dialogs: [],
        originals: []
      };
    }
    
    shotGroups[shotNum].frames.push(frame.frameNumber);
    if (frame.description) shotGroups[shotNum].descriptions.push(frame.description);
    if (frame.dialog) shotGroups[shotNum].dialogs.push(frame.dialog);
    if (frame.original) shotGroups[shotNum].originals.push(frame.original);
  }
  
  return Object.values(shotGroups).map(group => ({
    shotNumber: group.shotNumber,
    frames: group.frames,
    images: [], // No image cropping for now
    description: group.descriptions.join('\n'),
    dialog: group.dialogs.join('\n'),
    original: group.originals.join('\n')
  }));
}

/**
 * Storyboard extraction endpoint
 */
app.post('/api/extract-storyboard', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No PDF file uploaded' });
    }
    
    if (!process.env.ANTHROPIC_API_KEY) {
      return res.status(500).json({ error: 'ANTHROPIC_API_KEY not configured' });
    }
    
    console.log('[Storyboard] Processing PDF:', req.file.originalname, 'Size:', req.file.size);
    
    // Use Claude's direct PDF support
    const extracted = await extractWithDirectPdf(req.file.buffer);
    
    console.log(`[Storyboard] Extracted ${extracted.spots?.length || 0} spots`);
    
    // Group frames into shots
    const spots = (extracted.spots || []).map(spot => ({
      name: spot.name || 'Untitled Spot',
      shots: groupFramesIntoShots(spot.frames || [])
    }));
    
    console.log(`[Storyboard] Returning ${spots.length} spots`);
    
    res.json({ spots });
    
  } catch (error) {
    console.error('[Storyboard] Error:', error);
    res.status(500).json({ error: error.message || 'Extraction failed' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Railway PDF Server listening on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'production'}`);
  console.log(`Storyboard extraction: ${process.env.ANTHROPIC_API_KEY ? 'enabled' : 'disabled (no API key)'}`);
});
