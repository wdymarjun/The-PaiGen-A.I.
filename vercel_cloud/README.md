# PaiGen AI — Vercel Cloud Relay Server

Hosts PaiGen's online control plane on Vercel so you can control PaiGen and your Windows PC from **both your PC and your Vivo V29 mobile app over 5G/Wi-Fi anywhere**.

## 1. Deploy to Vercel (Free Hobby Tier)
1. Go to [vercel.com/new](https://vercel.com/new) and import `wdymarjun/The-PaiGen-A.I.`.
2. Set **Root Directory** to `vercel_cloud`.
3. Add **Environment Variables**:
   - `PAIGEN_CLOUD_TOKEN`: A strong secret passphrase you choose (e.g., `arjun-paigen-secret-2026`).
   - *(Optional for persistent state across serverless instances — recommended free tier)* Add Upstash Redis from Vercel Storage/Marketplace: `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`.
   - *(Optional so AI replies even when your PC is turned off)* `GEMINI_API_KEY` or `GROQ_API_KEY`.
4. Click **Deploy** — you get a permanent URL like `https://paigen-ai.vercel.app`.

## 2. Connect Your PC (`run_cloud_uplink.bat`)
On your Windows PC:
```powershell
set PAIGEN_CLOUD_URL=https://paigen-ai.vercel.app
set PAIGEN_CLOUD_TOKEN=arjun-paigen-secret-2026
run_cloud_uplink.bat
```
Your PC makes outbound HTTPS polls to Vercel—no router port forwarding required.

## 3. Connect Your Vivo V29 APK
Open **PaiGen AI** -> **MIND** tab:
- Set **PC / Cloud URL** to `https://paigen-ai.vercel.app`
- Set **Cloud Secret Token** to `arjun-paigen-secret-2026`
- Tap **Save Settings & Connect**.
