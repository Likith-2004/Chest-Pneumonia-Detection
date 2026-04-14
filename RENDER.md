# Render Deployment Quick Start

## Deploy to Render.com in 5 Steps 🚀

### Step 1: Sign Up on Render
1. Go to [render.com](https://render.com)
2. Sign up with GitHub account
3. Connect your GitHub repository

### Step 2: Create Web Service
1. Click **New → Web Service**
2. Select the repository: `Chest-X-Ray-Pneumonia-Detection`
3. Click **Connect**

### Step 3: Configure Settings
- **Name**: `pneumonia-detector`
- **Environment**: `Python`
- **Branch**: `main`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn -w 4 -b 0.0.0.0:$PORT web_app:app`
- **Instance Type**: `Free` (optional, pay as you go available)

### Step 4: Add Environment Variables (Optional)
- `FLASK_ENV`: `production`
- `FLASK_DEBUG`: `false`
- `USE_GPU`: `false`

### Step 5: Deploy
1. Click **Create Web Service**
2. Wait 2-3 minutes for deployment
3. Your app URL will appear: `https://pneumonia-detector-xxxxx.onrender.com`

## ✅ What You Get
- ✅ Free SSL certificate (HTTPS)
- ✅ Auto-deploys on every push to `main` branch
- ✅ Live URL to share
- ✅ Logs accessible from dashboard
- ✅ 1 free deployment

## 📊 App Details
- **Python Version**: 3.10+
- **Framework**: Flask
- **Model**: ResNet18 + Grad-CAM
- **Max Upload**: 16MB
- **Processing Device**: CPU (GPU free tier limited)

## 🔗 Your Deployment URL
After deployment, access at: `https://pneumonia-detector-xxxxx.onrender.com`

## 🔄 Auto-Deploy
Every time you push to GitHub (main branch), Render automatically redeploys your app!

```bash
# Make changes locally
git add .
git commit -m "Feature update"
git push origin main

# ✅ Automatically deployed to Render!
```

## 📝 File Structure (Render-Ready)
```
├── web_app.py              ⭐ Main Flask app
├── config.py               ⚙️ Configuration
├── pneumonia_unknown_model.pth  🧠 Model weights
├── requirements.txt        📋 Dependencies
├── Procfile               🚀 Render config
├── render.yaml            🎯 Optional config
├── templates/
│   └── index.html        🎨 All-in-one web interface
├── README.md             📖 Documentation
└── .gitignore            🔒 Git ignore rules
```

## 🆘 Troubleshooting

**App fails to deploy?**
- Check `Procfile` syntax
- Ensure `requirements.txt` is up to date
- Model file must be in project root

**App crashes after deploy?**
- Check logs: Render dashboard → Logs tab
- Verify `web_app.py` runs locally: `python web_app.py`
- Check model file exists

**App too slow?**
- Free tier has limited resources
- Upgrade to paid plan for better performance
- GPU acceleration not included in free tier

## 📞 Support
- Render docs: https://render.com/docs
- Flask docs: https://flask.palletsprojects.com
- GitHub Issues: Check your repository

---

**Deployed successfully? 🎉 Share your URL!**
