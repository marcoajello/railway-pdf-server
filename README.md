# Railway PDF Server

Simple Express server with Puppeteer for generating PDFs from HTML.

## Deployment to Railway

### 1. Prerequisites
- GitHub account
- Railway account (sign up at railway.app)

### 2. Steps

**Option A: Deploy from GitHub (Recommended)**

1. Push this folder to a GitHub repository:
   ```bash
   cd railway-pdf-server
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/railway-pdf-server.git
   git push -u origin main
   ```

2. Go to railway.app and click "New Project"

3. Select "Deploy from GitHub repo"

4. Select your `railway-pdf-server` repository

5. Railway will automatically:
   - Detect it's a Node.js app
   - Install dependencies
   - Run `npm start`
   - Give you a public URL

6. Copy the URL (looks like: `https://railway-pdf-server-production.up.railway.app`)

**Option B: Deploy from CLI**

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login:
   ```bash
   railway login
   ```

3. Deploy:
   ```bash
   cd railway-pdf-server
   railway init
   railway up
   ```

4. Get your URL:
   ```bash
   railway domain
   ```

### 3. Test Your Deployment

Once deployed, test with:
```bash
curl https://YOUR-RAILWAY-URL.up.railway.app/
```

Should return:
```json
{"status":"ok","message":"Railway PDF Server running","version":"1.0.0"}
```

### 4. Update Your Print Button

In your `script.js`, change the URL from Vercel to your Railway URL:

```javascript
const response = await fetch('https://YOUR-RAILWAY-URL.up.railway.app/generate-pdf', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ html, orientation: 'landscape' })
});
```

## How It Works

1. Your browser sends HTML to Railway server
2. Server launches Puppeteer (headless Chrome)
3. Chrome renders the HTML
4. Chrome generates PDF
5. Server sends PDF back to browser
6. Browser downloads the file

## Cost

- Free for 500 hours/month (plenty for this use case)
- After that: ~$5/month for hobby plan

## Troubleshooting

**If deployment fails:**
- Check Railway logs in the dashboard
- Ensure Node version is 18+
- Make sure all files are committed to Git

**If PDF generation fails:**
- Check that HTML is valid
- Look at Railway logs for error messages
- Test with simple HTML first

## API Endpoints

### GET /
Health check. Returns server status.

### POST /generate-pdf
Generate PDF from HTML.

**Request body:**
```json
{
  "html": "<html>...</html>",
  "orientation": "landscape"
}
```

**Response:**
PDF file (application/pdf)
