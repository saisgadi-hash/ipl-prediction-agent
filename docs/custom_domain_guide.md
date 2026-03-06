# Custom Domain Setup Guide

How to get a professional URL like `iplpredictor.com` instead of `yourapp.streamlit.app`.

## Option 1: Streamlit Cloud Custom Domain (Easiest)

Streamlit Cloud supports custom domains on their free tier:

1. **Buy a domain** from a registrar (Namecheap, Google Domains, Cloudflare — around $10-15/year)
2. **Deploy your app** to Streamlit Cloud (you've already done this!)
3. **In Streamlit Cloud dashboard**, go to your app → Settings → Custom Domain
4. **Add your domain** (e.g., `iplpredictor.com`)
5. **Add a CNAME record** in your domain registrar's DNS settings:
   - Type: `CNAME`
   - Host: `@` or `www`
   - Value: `<your-app>.streamlit.app`
6. **Wait 5-30 minutes** for DNS propagation
7. Streamlit automatically provisions an SSL certificate (https)

## Option 2: Cloudflare Tunnel (Free, More Control)

If you want to host on your own machine:

1. Sign up at [Cloudflare](https://cloudflare.com) (free plan)
2. Add your domain to Cloudflare
3. Install `cloudflared` on your server
4. Create a tunnel: `cloudflared tunnel create ipl-predictor`
5. Route traffic: `cloudflared tunnel route dns ipl-predictor iplpredictor.com`
6. Run: `cloudflared tunnel run ipl-predictor`

## Recommended Domain Names

- `iplpredictor.com` / `.in`
- `iplai.in`
- `cricketpredictions.com`
- `iplwinner.ai`

## Cost Summary

| Item | Cost |
|------|------|
| Domain (.com) | ~$10-15/year |
| Domain (.in) | ~$5-8/year |
| Streamlit Cloud hosting | Free |
| SSL Certificate | Free (auto) |
| **Total** | **$5-15/year** |
