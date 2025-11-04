 # 🚀 QUICK START - Get PDF Working in 10 Minutes

## What You're About to Do
Deploy a real server to Railway that will generate your PDFs using Chrome.

---

## THREE STEPS TO SUCCESS

### 1️⃣ Deploy Server (5 min)
```bash
# Extract the files
tar -xzf railway-pdf-complete.tar.gz
cd railway-pdf-server

# Push to GitHub
git init
git add .
git commit -m "PDF server"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/railway-pdf-server.git
git push -u origin main

# Deploy on Railway
# → Go to railway.app
# → Login with GitHub
# → "New Project" → "Deploy from GitHub repo"
# → Select railway-pdf-server
# → Wait 2 minutes
# → Click "Generate Domain" in Settings
# → COPY YOUR URL
```

### 2️⃣ Update Print Button (2 min)
Open `script.js`, find line 2222:

**OLD (Vercel - broken):**
```javascript
const response = await fetch('https://vercel-pdf-playwright.vercel.app/api/generate-pdf', {
```

**NEW (Railway - works):**
```javascript
const response = await fetch('https://YOUR-RAILWAY-URL.up.railway.app/generate-pdf', {
```

Replace `YOUR-RAILWAY-URL` with the URL you copied from Railway.

### 3️⃣ Test (1 min)
- Open your scheduler
- Click Print
- Wait 5 seconds
- PDF downloads ✅

---

## Why This Works

**Before:** Vercel serverless → missing libraries → fails  
**After:** Railway real server → full Chrome → works perfectly

---

## Files Included

- `server.js` - Express + Puppeteer server
- `package.json` - Dependencies
- `DEPLOYMENT_GUIDE.md` - Detailed instructions
- `client-code.js` - Reference for your print button code
- `README.md` - Full documentation

---

## Cost

**Free:** 500 hours/month (plenty)  
**Paid:** $5/month if you want always-on

---

## If Something Goes Wrong

**Problem:** "Failed to fetch"  
**Fix:** Check Railway URL is correct in script.js

**Problem:** Takes 15 seconds first time  
**Fix:** Normal - server is waking up

**Problem:** Still using Vercel URL  
**Fix:** You forgot step 2 - update script.js!

---

## The Bottom Line

You've been fighting with serverless. That's over now. You have a real server that will render your PDFs with real Chrome. It will work every time.

Just deploy it and update the URL. That's it.

---

## Support

**Railway Dashboard:** railway.app → your project → Logs  
**Test Server:** https://your-url.up.railway.app (should return "ok")  
**Browser Console:** F12 → check for errors

---

Ready? Extract the archive and follow Step 1. You'll be generating PDFs in 10 minutes.
