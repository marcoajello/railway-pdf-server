# Quick Deployment Guide - Railway PDF Server

## Step 1: Deploy to Railway (5 minutes)

### Option A: GitHub (Easiest)

1. **Create GitHub repo:**
   - Go to github.com/new
   - Name it `railway-pdf-server`
   - Create repository

2. **Push code to GitHub:**
   ```bash
   cd railway-pdf-server
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/railway-pdf-server.git
   git push -u origin main
   ```

3. **Deploy on Railway:**
   - Go to railway.app
   - Click "Login" → Sign up with GitHub
   - Click "New Project"
   - Click "Deploy from GitHub repo"
   - Select `railway-pdf-server`
   - Wait 2-3 minutes for deployment

4. **Get your URL:**
   - Click "Settings" tab
   - Click "Generate Domain"
   - Copy the URL (e.g., `railway-pdf-server-production.up.railway.app`)

### Option B: Railway CLI (Alternative)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
cd railway-pdf-server
railway init
railway up

# Get URL
railway domain
```

---

## Step 2: Update Your Print Button (2 minutes)

1. **Open your `script.js` file**

2. **Find line 2222** (the old Vercel URL):
   ```javascript
   const response = await fetch('https://vercel-pdf-playwright.vercel.app/api/generate-pdf', {
   ```

3. **Replace with your Railway URL:**
   ```javascript
   const response = await fetch('https://YOUR-RAILWAY-URL.up.railway.app/generate-pdf', {
   ```
   
   For example:
   ```javascript
   const response = await fetch('https://railway-pdf-server-production.up.railway.app/generate-pdf', {
   ```

4. **Save the file**

---

## Step 3: Test It (1 minute)

1. Open your scheduler app in browser
2. Click the "Print" button
3. Wait 3-5 seconds
4. PDF should download automatically

**First run might take 10-15 seconds** while Railway spins up the server. After that, it's fast.

---

## Troubleshooting

### "Failed to generate PDF: Failed to fetch"
- Check that Railway server is running (go to railway.app dashboard)
- Verify you copied the URL correctly
- Make sure URL starts with `https://` not `http://`

### "Failed to generate PDF: Load failed"
- This was the OLD Vercel error
- Make sure you updated the URL in script.js
- Clear your browser cache and reload

### "PDF is blank or broken"
- Check Railway logs in dashboard for errors
- Verify images are loading (check browser console)

### Server goes to sleep
- Railway free tier: server sleeps after 30 minutes of no use
- First request after sleep takes 10-15 seconds
- Upgrade to Hobby plan ($5/mo) for always-on server

---

## What You Just Built

```
┌─────────────┐
│   Browser   │  User clicks "Print"
│  (Your App) │
└──────┬──────┘
       │ Sends HTML
       │ (with all images as data URLs)
       ▼
┌─────────────────────┐
│  Railway Server     │  Puppeteer renders HTML
│  (Node + Puppeteer) │  Generates PDF
└──────┬──────────────┘
       │ Returns PDF file
       ▼
┌─────────────┐
│   Browser   │  Downloads PDF
│  (Your App) │
└─────────────┘
```

**Key advantages:**
- Runs on real Linux server with full Chrome
- No serverless limitations
- Works with complex layouts
- Handles images perfectly
- No pagination issues

---

## Cost

- **Free tier:** 500 hours/month, server sleeps after 30 min
- **Hobby plan:** $5/month, always-on, faster

For your use case (occasional PDF generation), free tier is plenty.

---

## Next Steps

Once it works:
1. Test with different table layouts
2. Try adding more rows/images
3. Check pagination on multi-page PDFs
4. Tweak CSS in `buildCompleteHTML()` if needed

---

## Support

If something doesn't work:
1. Check Railway logs (dashboard → Deployments → Logs)
2. Check browser console for errors
3. Verify the URL is correct in script.js

Common fixes:
- Wait 30 seconds and try again (server might be starting)
- Hard refresh browser (Cmd+Shift+R or Ctrl+Shift+R)
- Check that Railway deployment succeeded (green checkmark)
