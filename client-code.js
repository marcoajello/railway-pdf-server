// ===========================================
// RAILWAY PDF CLIENT CODE
// ===========================================
// Replace this in your script.js after deploying to Railway
// 
// INSTRUCTIONS:
// 1. Deploy railway-pdf-server to Railway (see README.md)
// 2. Get your Railway URL (e.g., https://your-app.up.railway.app)
// 3. Replace 'YOUR_RAILWAY_URL_HERE' below with your actual URL
// 4. Replace the old generatePDFWithPuppeteer function in script.js
// ===========================================

// *** CONFIGURATION - UPDATE THIS AFTER DEPLOYMENT ***
const RAILWAY_PDF_SERVER = 'YOUR_RAILWAY_URL_HERE'; // e.g., 'https://railway-pdf-server-production.up.railway.app'

// Attach to print button (this line should already exist in your script.js around line 2211)
// printBtn && printBtn.addEventListener('click', generatePDFWithPuppeteer);

async function generatePDFWithPuppeteer() {
  try {
    const btn = document.getElementById('printBtn');
    const originalText = btn.textContent;
    btn.textContent = 'Generating PDF...';
    btn.disabled = true;

    // Build complete HTML with all images converted to data URLs
    const html = await buildCompleteHTML();
    
    // Send to Railway server
    const response = await fetch(`${RAILWAY_PDF_SERVER}/generate-pdf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        html: html,
        orientation: 'landscape' 
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage = 'PDF generation failed';
      try {
        const errorJson = JSON.parse(errorText);
        errorMessage = errorJson.message || errorJson.error || errorMessage;
      } catch (e) {
        errorMessage = errorText || errorMessage;
      }
      throw new Error(errorMessage);
    }

    // Download the PDF
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `schedule-${new Date().toISOString().split('T')[0]}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    btn.textContent = originalText;
    btn.disabled = false;

  } catch (error) {
    console.error('PDF generation error:', error);
    alert('Failed to generate PDF: ' + error.message + '\n\nMake sure your Railway server is deployed and the URL is correct.');
    const btn = document.getElementById('printBtn');
    btn.textContent = 'Print';
    btn.disabled = false;
  }
}

// This function should already exist in your script.js (around line 2258)
// If it doesn't, add it:
async function buildCompleteHTML() {
  const metaTitle = document.getElementById('metaTitle')?.value || '';
  const metaVersion = document.getElementById('metaVersion')?.value || '';
  const metaDate = document.getElementById('metaDate')?.value || '';
  const metaDow = document.getElementById('metaDow')?.value || '';
  const metaX = document.getElementById('metaX')?.value || '';
  const metaY = document.getElementById('metaY')?.value || '';

  let metaDisplay = '';
  if (metaTitle) metaDisplay += metaTitle;
  if (metaVersion) metaDisplay += ` - ${metaVersion}`;
  if (metaDate) metaDisplay += ` - ${metaDate}`;
  if (metaDow) metaDisplay += ` (${metaDow})`;
  if (metaX && metaY) metaDisplay += ` - Day ${metaX} of ${metaY}`;

  // Clone the table
  const originalTable = document.getElementById('scheduleTable');
  const tableClone = originalTable.cloneNode(true);

  // Process all cells to extract values from inputs
  const cells = tableClone.querySelectorAll('td, th');
  for (const cell of cells) {
    const input = cell.querySelector('input, textarea, select');
    if (input) {
      const value = input.value || '';
      cell.textContent = value;
    }
  }

  // Convert blob URLs to data URLs for images
  const images = tableClone.querySelectorAll('img');
  for (const img of images) {
    if (img.src.startsWith('blob:')) {
      try {
        const dataUrl = await blobToDataURL(img.src);
        img.src = dataUrl;
      } catch (error) {
        console.error('Failed to convert image:', error);
        img.remove();
      }
    }
  }

  // Get all styles
  const styles = Array.from(document.styleSheets)
    .map(sheet => {
      try {
        return Array.from(sheet.cssRules || [])
          .map(rule => rule.cssText)
          .join('\n');
      } catch (e) {
        return '';
      }
    })
    .join('\n');

  // Build complete HTML
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        ${styles}
        
        /* Print-specific overrides */
        body {
          margin: 0;
          padding: 20px;
          background: white;
        }
        
        table {
          width: 100%;
          border-collapse: collapse;
        }
        
        th, td {
          border: 1px solid #ddd;
          padding: 8px;
          text-align: left;
          font-size: 10pt;
        }
        
        th {
          background-color: #f2f2f2;
          font-weight: bold;
        }
        
        img {
          max-width: 100px;
          max-height: 100px;
          object-fit: contain;
        }
        
        .meta-display {
          font-size: 14pt;
          font-weight: bold;
          margin-bottom: 20px;
          padding: 10px;
          background: #f5f5f5;
          border-radius: 4px;
        }
      </style>
    </head>
    <body>
      ${metaDisplay ? `<div class="meta-display">${metaDisplay}</div>` : ''}
      ${tableClone.outerHTML}
    </body>
    </html>
  `;

  return html;
}

async function blobToDataURL(blobUrl) {
  const response = await fetch(blobUrl);
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}
