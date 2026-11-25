# PaiGen AI — Render & Vercel Cloud Relay Server (with Live Logs)

Hosts PaiGen's online server and structured Logs system (`GET /api/logs` + stdout Render stream) so you can control PaiGen and your Windows PC from **both your PC and your Vivo V29 mobile APK anywhere**.

## Option A: Deploy to Render (Recommended — Free Tier, Zero Redis Needed)
1. Click the 1-Click Render Blueprint link for this repo or open [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints) and connect `wdymarjun/The-PaiGen-A.I.`.
2. Render automatically reads `render.yaml` (`paigen-ai-cloud`, Python, free tier, `/api/ping` health check, auto-generated `PAIGEN_CLOUD_TOKEN`).
3. Watch your live server logs directly in **Render Dashboard -> Logs** AND inside the app's **Live Server, PC & Call Logs** panel (`/api/logs`).
4. *(Optional)* Add `GEMINI_API_KEY` or `GROQ_API_KEY` in Render Environment Variables if you want AI replies even while your PC is turned off.

## Option B: Deploy to Vercel
1. Go to [vercel.com/new](https://vercel.com/new), import `wdymarjun/The-PaiGen-A.I.`, set **Root Directory = `vercel_cloud`**, and add `PAIGEN_CLOUD_TOKEN` + optional Upstash Redis keys.

## Connect Your PC (`run_cloud_uplink.bat`)
```powershell
set PAIGEN_CLOUD_URL=https://paigen-ai-cloud.onrender.com
set PAIGEN_CLOUD_TOKEN=<your-render-token>
run_cloud_uplink.bat
```

## Connect Your Vivo V29 APK
Open **PaiGen AI** -> **MIND** tab -> enter your `https://...onrender.com` URL and `PAIGEN_CLOUD_TOKEN` -> **Save Settings & Connect**.
