# Giga-World-1-projectpage

🔗 Live page: <https://yvonne-oh.github.io/Giga-World-1-projectpage>

## Website Structure

```
github_page/
├── index.html        # The full project page
├── style.css         # All styling
├── assets/           # Brand assets (logos, hero artwork)
├── images/           # Figures, charts, teaser images
│   ├── teaser/       # Hero teaser image
│   ├── model/        # Architecture / trajectory figures
│   ├── exp/          # Closed-loop correlation figures
│   ├── metric/       # Metric heatmaps
│   └── accel/        # Acceleration benchmark SVGs
└── video/            # All video content
    ├── video_Wall/   # Background hero video wall
    ├── control_gif/  # Multi-view control GIF grid
    ├── cc/           # Closed-loop rollout episodes
    ├── ood/          # Out-of-distribution rollouts
    ├── model_trans/  # Transfer & conditioning
    └── flash_and_ultra_gen/  # Flash inference demos
```

## Local Preview

The project page is a static site — open it with any HTTP server:

```bash
# from the repo root
python3 -m http.server 8765
# then open http://localhost:8765
```

## Deploy to GitHub Pages

The site is published from the `main` branch root.

1. Push to GitHub:
   ```bash
   git push origin main
   ```
2. On GitHub, open **Settings → Pages** for the repository.
3. Under **Source**, choose **Deploy from a branch**.
4. Set **Branch** to `main` and folder to `/ (root)`, then **Save**.
5. Wait ~1 minute for the first deploy. The page will be live at:
   ```
   https://yvonne-oh.github.io/Giga-World-1-projectpage/
   ```