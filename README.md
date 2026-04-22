# <!DOCTYPE html>

# <html lang="en">

# <head>

# <meta charset="UTF-8"/>

# <meta name="viewport" content="width=device-width,initial-scale=1.0"/>

# <title>Shortcut Learning Detector · Docs</title>

# <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue\&family=Outfit:wght@300;400;500;600;700\&family=JetBrains+Mono:wght@300;400;500\&display=swap" rel="stylesheet"/>

# <style>

# :root{

# &#x20; --bg:#04050a;

# &#x20; --s1:#080c14;

# &#x20; --s2:#0d1220;

# &#x20; --s3:#131928;

# &#x20; --border:rgba(255,255,255,0.06);

# &#x20; --border2:rgba(255,255,255,0.12);

# &#x20; --cyan:#00e5ff;

# &#x20; --orange:#ff6b2b;

# &#x20; --green:#00e5a0;

# &#x20; --red:#ff3d6b;

# &#x20; --yellow:#ffd84d;

# &#x20; --text:#dde4f0;

# &#x20; --muted:#5a6680;

# &#x20; --muted2:#8895b0;

# }

# \*{box-sizing:border-box;margin:0;padding:0}

# html{scroll-behavior:smooth}

# body{background:var(--bg);color:var(--text);font-family:'Outfit',sans-serif;font-weight:400;overflow-x:hidden;cursor:none}

# 

# .cursor{position:fixed;width:10px;height:10px;background:var(--cyan);border-radius:50%;pointer-events:none;z-index:9999;transform:translate(-50%,-50%);mix-blend-mode:difference;transition:width .2s,height .2s}

# .cursor-ring{position:fixed;width:36px;height:36px;border:1.5px solid rgba(0,229,255,0.4);border-radius:50%;pointer-events:none;z-index:9998;transform:translate(-50%,-50%)}

# 

# body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");opacity:.025;pointer-events:none;z-index:1}

# 

# .bg-canvas{position:fixed;inset:0;z-index:0;background:radial-gradient(ellipse 70% 50% at 80% 10%,rgba(0,229,255,.06) 0,transparent 60%),radial-gradient(ellipse 50% 60% at 10% 80%,rgba(255,107,43,.05) 0,transparent 60%),radial-gradient(ellipse 40% 40% at 50% 50%,rgba(0,229,160,.03) 0,transparent 60%)}

# .scan-line{position:fixed;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.25;z-index:2;animation:scan 7s linear infinite}

# @keyframes scan{0%{top:-2px}100%{top:100vh}}

# 

# .page{position:relative;z-index:3;max-width:1000px;margin:0 auto;padding:0 28px 120px}

# 

# /\* HERO \*/

# .hero{min-height:100vh;display:flex;flex-direction:column;justify-content:center;padding:60px 0 80px;position:relative}

# .hero-eyebrow{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--cyan);letter-spacing:.15em;text-transform:uppercase;margin-bottom:28px;opacity:0;animation:fadeUp .6s .1s ease forwards}

# .hero-eyebrow::before,.hero-eyebrow::after{content:'';width:32px;height:1px;background:var(--cyan);opacity:.6}

# .hero-title{font-family:'Bebas Neue',sans-serif;font-size:clamp(4rem,11vw,9rem);line-height:.92;letter-spacing:.02em;margin-bottom:8px;opacity:0;animation:fadeUp .7s .2s ease forwards}

# .hero-title .line1{display:block;color:var(--text)}

# .hero-title .line2{display:block;-webkit-text-stroke:1.5px var(--cyan);color:transparent;position:relative}

# .hero-title .line2::after{content:attr(data-text);position:absolute;left:0;top:0;color:var(--cyan);clip-path:polygon(0 0,0 0,0 100%,0 100%);animation:reveal-text 1.4s .9s cubic-bezier(.77,0,.18,1) forwards}

# @keyframes reveal-text{to{clip-path:polygon(0 0,100% 0,100% 100%,0 100%)}}

# .hero-title .line3{display:block;color:var(--cyan)}

# .hero-sub{font-size:1rem;color:var(--muted2);font-weight:300;max-width:560px;line-height:1.8;margin:28px 0 40px;opacity:0;animation:fadeUp .6s .5s ease forwards}

# .hero-meta{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:44px;opacity:0;animation:fadeUp .6s .65s ease forwards}

# .meta-chip{display:flex;align-items:center;gap:8px;background:var(--s2);border:1px solid var(--border2);border-radius:100px;padding:8px 16px;font-size:.8rem;transition:border-color .3s,background .3s}

# .meta-chip:hover{border-color:rgba(0,229,255,.3);background:var(--s3)}

# .meta-chip .dot{width:6px;height:6px;border-radius:50%;background:var(--cyan);box-shadow:0 0 8px var(--cyan)}

# .hero-links{display:flex;gap:14px;flex-wrap:wrap;opacity:0;animation:fadeUp .6s .8s ease forwards}

# .btn{display:inline-flex;align-items:center;gap:10px;padding:13px 26px;border-radius:10px;font-size:.88rem;font-weight:600;text-decoration:none;transition:all .25s;position:relative;overflow:hidden}

# .btn::before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);transform:translateX(-100%);transition:transform .4s}

# .btn:hover::before{transform:translateX(100%)}

# .btn-primary{background:var(--cyan);color:#04050a;box-shadow:0 0 24px rgba(0,229,255,.25)}

# .btn-primary:hover{background:#33ecff;box-shadow:0 0 40px rgba(0,229,255,.45);transform:translateY(-2px)}

# .btn-secondary{background:transparent;color:var(--text);border:1px solid var(--border2)}

# .btn-secondary:hover{border-color:var(--cyan);color:var(--cyan);transform:translateY(-2px)}

# .scroll-hint{position:absolute;bottom:40px;left:0;display:flex;align-items:center;gap:12px;font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--muted);opacity:0;animation:fadeUp .6s 1.2s ease forwards}

# .scroll-line{width:48px;height:1px;background:var(--muted);position:relative;overflow:hidden}

# .scroll-line::after{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:var(--cyan);animation:scroll-pulse 2s 1.5s ease-in-out infinite}

# @keyframes scroll-pulse{0%{left:-100%}100%{left:100%}}

# 

# /\* DIVIDER \*/

# .divider{display:flex;align-items:center;gap:16px;margin:80px 0 48px}

# .divider-line{flex:1;height:1px;background:var(--border)}

# .divider-label{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;white-space:nowrap}

# .divider-num{font-family:'Bebas Neue',sans-serif;font-size:.9rem;color:var(--cyan);letter-spacing:.05em}

# 

# /\* SECTION TITLE \*/

# .section-title{font-family:'Bebas Neue',sans-serif;font-size:clamp(1.8rem,4vw,2.6rem);letter-spacing:.04em;margin-bottom:32px}

# .section-title .accent{color:var(--cyan)}

# 

# /\* CARDS \*/

# .card{background:var(--s1);border:1px solid var(--border);border-radius:16px;padding:28px 32px;margin-bottom:16px;transition:border-color .3s,transform .3s,box-shadow .3s;position:relative;overflow:hidden}

# .card::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(0,229,255,.03),transparent 50%);opacity:0;transition:opacity .3s}

# .card:hover{border-color:rgba(0,229,255,.2);transform:translateY(-3px);box-shadow:0 12px 40px rgba(0,0,0,.4)}

# .card:hover::before{opacity:1}

# .card-title{font-weight:700;font-size:1.05rem;margin-bottom:10px}

# .card-body{font-size:.88rem;color:var(--muted2);line-height:1.75}

# 

# /\* MODEL COMPARISON \*/

# .model-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:24px 0}

# .model-card{border-radius:14px;padding:24px;position:relative;overflow:hidden}

# .model-card.biased{background:linear-gradient(135deg,rgba(255,61,107,.08),rgba(255,107,43,.05));border:1px solid rgba(255,61,107,.25)}

# .model-card.unbiased{background:linear-gradient(135deg,rgba(0,229,160,.08),rgba(0,229,255,.05));border:1px solid rgba(0,229,160,.25)}

# .model-card::after{content:'';position:absolute;top:-40px;right:-40px;width:120px;height:120px;border-radius:50%;opacity:.06}

# .model-card.biased::after{background:var(--red)}

# .model-card.unbiased::after{background:var(--green)}

# .model-badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.7rem;font-weight:500;padding:4px 12px;border-radius:100px;margin-bottom:14px;letter-spacing:.08em;text-transform:uppercase}

# .biased .model-badge{background:rgba(255,61,107,.15);color:var(--red)}

# .unbiased .model-badge{background:rgba(0,229,160,.12);color:var(--green)}

# .model-name{font-weight:700;font-size:1rem;margin-bottom:10px}

# .model-body{font-size:.83rem;color:var(--muted2);line-height:1.7}

# 

# /\* TECH GRID \*/

# .tech-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:8px}

# .tech-chip{background:var(--s2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;transition:border-color .25s,background .25s}

# .tech-chip:hover{border-color:rgba(0,229,255,.25);background:var(--s3)}

# .tech-chip-cat{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}

# .tech-chip-name{font-weight:600;font-size:.85rem}

# 

# /\* STEPS \*/

# .steps{counter-reset:step-c}

# .step{display:flex;gap:22px;margin-bottom:28px;padding-bottom:28px;border-bottom:1px solid var(--border)}

# .step:last-child{border-bottom:none;margin-bottom:0;padding-bottom:0}

# .step-num{counter-increment:step-c;width:38px;height:38px;flex-shrink:0;border-radius:10px;background:var(--s3);border:1px solid var(--border2);display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--cyan);margin-top:3px}

# .step-num::before{content:counter(step-c,decimal-leading-zero)}

# .step-body h4{font-weight:600;font-size:.95rem;margin-bottom:8px}

# .step-body p{font-size:.86rem;color:var(--muted2);line-height:1.7;margin-bottom:0}

# 

# /\* CODE \*/

# pre{background:#020408;border:1px solid var(--border);border-left:3px solid var(--cyan);border-radius:12px;padding:18px 22px;font-family:'JetBrains Mono',monospace;font-size:.8rem;color:#a8d8ea;overflow-x:auto;margin:12px 0 16px;line-height:1.7}

# pre .cm{color:#3a5060}

# code{font-family:'JetBrains Mono',monospace;font-size:.82rem;background:rgba(0,229,255,.08);color:var(--cyan);padding:2px 8px;border-radius:5px}

# 

# /\* FLOW \*/

# .flow{display:flex;flex-wrap:wrap;align-items:center;gap:0;margin:20px 0}

# .flow-node{background:var(--s2);border:1px solid var(--border2);border-radius:8px;padding:9px 16px;font-size:.8rem;font-weight:500;white-space:nowrap;transition:border-color .25s}

# .flow-node:hover{border-color:var(--cyan);color:var(--cyan)}

# .flow-arrow{padding:0 8px;color:var(--muted);font-size:.9rem}

# 

# /\* TABLE \*/

# .tbl-wrap{border-radius:14px;overflow:hidden;border:1px solid var(--border);margin:16px 0}

# table{width:100%;border-collapse:collapse;font-size:.85rem}

# thead tr{background:var(--s2)}

# th{padding:12px 18px;text-align:left;font-weight:600;font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid var(--border)}

# td{padding:13px 18px;border-bottom:1px solid var(--border);color:var(--muted2);vertical-align:middle}

# tr:last-child td{border-bottom:none}

# tr:hover td{background:rgba(255,255,255,.02)}

# 

# /\* PILLS \*/

# .pill{display:inline-block;padding:3px 11px;border-radius:100px;font-size:.73rem;font-weight:600;font-family:'JetBrains Mono',monospace}

# .pill-cyan{background:rgba(0,229,255,.1);color:var(--cyan)}

# .pill-green{background:rgba(0,229,160,.1);color:var(--green)}

# .pill-red{background:rgba(255,61,107,.1);color:var(--red)}

# .pill-orange{background:rgba(255,107,43,.1);color:var(--orange)}

# .pill-yellow{background:rgba(255,216,77,.1);color:var(--yellow)}

# 

# /\* ALERTS \*/

# .alert{border-radius:12px;padding:16px 20px;display:flex;gap:14px;align-items:flex-start;font-size:.86rem;line-height:1.7;margin:16px 0}

# .alert-icon{font-size:1rem;flex-shrink:0;margin-top:1px}

# .alert.warn{background:rgba(255,216,77,.06);border:1px solid rgba(255,216,77,.2);color:#ffe08a}

# .alert.info{background:rgba(0,229,255,.05);border:1px solid rgba(0,229,255,.18);color:#7de8ff}

# .alert.success{background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.2);color:#7dffc8}

# .alert.danger{background:rgba(255,61,107,.06);border:1px solid rgba(255,61,107,.2);color:#ff9db8}

# 

# /\* TROUBLESHOOT \*/

# .trouble{border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:12px;transition:border-color .25s}

# .trouble:hover{border-color:rgba(0,229,255,.2)}

# .trouble-q{padding:16px 20px;background:var(--s1);display:flex;align-items:center;gap:12px;font-weight:600;font-size:.88rem}

# .trouble-tag{font-family:'JetBrains Mono',monospace;font-size:.72rem;background:rgba(255,61,107,.12);color:var(--red);padding:3px 10px;border-radius:6px;flex-shrink:0}

# .trouble-a{padding:16px 20px;background:var(--s2);font-size:.85rem;color:var(--muted2);line-height:1.7;border-top:1px solid var(--border)}

# 

# /\* PIPELINE \*/

# .pipe-stage{display:flex;gap:16px;align-items:flex-start;padding:18px 20px;background:var(--s1);border:1px solid var(--border);border-radius:12px;margin-bottom:8px;transition:border-color .3s,background .3s}

# .pipe-stage:hover{border-color:rgba(0,229,255,.2);background:var(--s2)}

# .pipe-num{font-family:'Bebas Neue',sans-serif;font-size:1.6rem;color:rgba(0,229,255,.2);line-height:1;flex-shrink:0;width:28px;transition:color .3s}

# .pipe-stage:hover .pipe-num{color:var(--cyan)}

# .pipe-content h4{font-weight:600;font-size:.92rem;margin-bottom:4px}

# .pipe-content p{font-size:.83rem;color:var(--muted2)}

# 

# /\* TEST CARDS \*/

# .test-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;margin-top:8px}

# .test-card{background:var(--s2);border:1px solid var(--border);border-radius:12px;padding:18px 20px;display:flex;gap:14px;align-items:flex-start;transition:border-color .3s}

# .test-card:hover{border-color:rgba(0,229,160,.3)}

# .test-check{width:26px;height:26px;flex-shrink:0;background:rgba(0,229,160,.1);border-radius:6px;display:flex;align-items:center;justify-content:center;color:var(--green);font-size:.85rem}

# .test-card h4{font-weight:600;font-size:.88rem;margin-bottom:4px}

# .test-card p{font-size:.81rem;color:var(--muted2);line-height:1.6}

# 

# /\* FOOTER \*/

# footer{margin-top:100px;padding:40px 0 60px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px}

# .footer-brand{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:.05em;color:var(--muted)}

# .footer-brand span{color:var(--cyan)}

# .footer-info{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--muted);text-align:right;line-height:1.8}

# 

# /\* ANIMATIONS \*/

# @keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}

# .reveal{opacity:0;transform:translateY(30px);transition:opacity .7s ease,transform .7s ease}

# .reveal.visible{opacity:1;transform:translateY(0)}

# 

# ::-webkit-scrollbar{width:5px;height:5px}

# ::-webkit-scrollbar-track{background:transparent}

# ::-webkit-scrollbar-thumb{background:var(--s3);border-radius:10px}

# ::-webkit-scrollbar-thumb:hover{background:rgba(0,229,255,.2)}

# 

# @media(max-width:680px){

# &#x20; .page{padding:0 18px 80px}

# &#x20; .model-grid{grid-template-columns:1fr}

# &#x20; .hero-title{font-size:clamp(3rem,13vw,5rem)}

# &#x20; footer{flex-direction:column;align-items:flex-start}

# &#x20; .footer-info{text-align:left}

# }

# </style>

# </head>

# <body>

# 

# <div class="cursor" id="cursor"></div>

# <div class="cursor-ring" id="cursor-ring"></div>

# <div class="bg-canvas"></div>

# <div class="scan-line"></div>

# 

# <div class="page">

# 

# &#x20; <!-- HERO -->

# &#x20; <section class="hero">

# &#x20;   <div class="hero-eyebrow">Project 45 \&nbsp;·\&nbsp; Team 85 \&nbsp;·\&nbsp; GLA University</div>

# &#x20;   <h1 class="hero-title">

# &#x20;     <span class="line1">Shortcut</span>

# &#x20;     <span class="line2" data-text="Learning">Learning</span>

# &#x20;     <span class="line3">Detector</span>

# &#x20;   </h1>

# &#x20;   <p class="hero-sub">An end-to-end deep learning diagnostic system that exposes and mitigates shortcut learning in CNNs — powered by PyTorch, FastAPI, and real-time Grad-CAM heatmap visualization.</p>

# &#x20;   <div class="hero-meta">

# &#x20;     <div class="meta-chip"><span class="dot"></span> Mayank \&nbsp;(Leader)</div>

# &#x20;     <div class="meta-chip"><span class="dot" style="background:var(--orange)"></span> Naitik Agarwal</div>

# &#x20;     <div class="meta-chip"><span class="dot" style="background:var(--green)"></span> Radhika Gupta</div>

# &#x20;     <div class="meta-chip" style="border-color:rgba(255,216,77,.2)"><span class="dot" style="background:var(--yellow)"></span> Mentor · Mr. Preshit Desai</div>

# &#x20;   </div>

# &#x20;   <div class="hero-links">

# &#x20;     <a href="https://shortcut-learning-detector.vercel.app/" class="btn btn-primary" target="\_blank">🌐 Live Frontend</a>

# &#x20;     <a href="https://shortcut-learning-detector-pgcc.onrender.com" class="btn btn-secondary" target="\_blank">⚙️ Live API \&nbsp;/docs</a>

# &#x20;   </div>

# &#x20;   <div class="scroll-hint"><div class="scroll-line"></div>scroll to explore</div>

# &#x20; </section>

# 

# &#x20; <!-- 01 OVERVIEW -->

# &#x20; <div class="divider reveal">

# &#x20;   <div class="divider-num">01</div><div class="divider-line"></div>

# &#x20;   <div class="divider-label">Project Overview</div><div class="divider-line"></div>

# &#x20; </div>

# 

# &#x20; <div class="reveal">

# &#x20;   <h2 class="section-title">What Is <span class="accent">Shortcut Learning?</span></h2>

# &#x20;   <div class="card">

# &#x20;     <p class="card-body">Deep learning models frequently "cheat" by latching onto <strong style="color:var(--text)">unintended statistical correlations</strong> instead of genuine semantic features. A CNN trained on Colored MNIST — where every digit "1" always appears on a green background — learns to associate <em>greenness</em> with the digit class, not the actual stroke shape. Change the background and the model collapses.<br><br>This system makes that invisible process <strong style="color:var(--text)">visible, measurable, and fixable.</strong></p>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <div class="model-grid reveal">

# &#x20;   <div class="model-card biased">

# &#x20;     <span class="model-badge">Biased</span>

# &#x20;     <div class="model-name">🔴 The Cheater Model</div>

# &#x20;     <p class="model-body">Trained on strictly color-correlated MNIST data. Learns background hues as the primary signal. Fails completely on unseen color combos. Grad-CAM highlights background edges — the digit is ignored.</p>

# &#x20;   </div>

# &#x20;   <div class="model-card unbiased">

# &#x20;     <span class="model-badge">Unbiased</span>

# &#x20;     <div class="model-name">🟢 The Fixed Model</div>

# &#x20;     <p class="model-body">Trained with randomized background augmentation. Forced to extract geometric stroke features. Classifies correctly regardless of background color. Grad-CAM illuminates the digit's stroke perfectly.</p>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <div class="reveal" style="margin-top:24px">

# &#x20;   <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:14px">End-to-End Flow</div>

# &#x20;   <div class="flow">

# &#x20;     <span class="flow-node">📤 Upload Image</span><span class="flow-arrow">→</span>

# &#x20;     <span class="flow-node">🔘 Select Model</span><span class="flow-arrow">→</span>

# &#x20;     <span class="flow-node">🌐 Axios POST</span><span class="flow-arrow">→</span>

# &#x20;     <span class="flow-node">⚙️ FastAPI Inference</span><span class="flow-arrow">→</span>

# &#x20;     <span class="flow-node">🔥 Grad-CAM</span><span class="flow-arrow">→</span>

# &#x20;     <span class="flow-node">📊 Dashboard</span>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <!-- 02 TECH STACK -->

# &#x20; <div class="divider reveal">

# &#x20;   <div class="divider-num">02</div><div class="divider-line"></div>

# &#x20;   <div class="divider-label">Tech Stack</div><div class="divider-line"></div>

# &#x20; </div>

# &#x20; <div class="reveal">

# &#x20;   <h2 class="section-title">Full <span class="accent">Architecture</span></h2>

# &#x20;   <div class="card">

# &#x20;     <div class="tech-grid">

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">Frontend</div><div class="tech-chip-name">React.js</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">HTTP Client</div><div class="tech-chip-name">Axios</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">UI Hosting</div><div class="tech-chip-name">Vercel Edge</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">API Framework</div><div class="tech-chip-name">FastAPI</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">Runtime</div><div class="tech-chip-name">Python 3.10</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">ASGI Server</div><div class="tech-chip-name">Uvicorn</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">ML Engine</div><div class="tech-chip-name">PyTorch</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">Vision</div><div class="tech-chip-name">Torchvision</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">Heatmaps</div><div class="tech-chip-name">OpenCV</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">Database</div><div class="tech-chip-name">SQLite + ORM</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">Object Store</div><div class="tech-chip-name">Cloudinary CDN</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">Caching</div><div class="tech-chip-name">Redis</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">MLOps</div><div class="tech-chip-name">Weights \& Biases</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">Container</div><div class="tech-chip-name">Docker</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">Testing</div><div class="tech-chip-name">PyTest + HTTPX</div></div>

# &#x20;       <div class="tech-chip"><div class="tech-chip-cat">CI / CD</div><div class="tech-chip-name">GitHub Actions</div></div>

# &#x20;     </div>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <!-- 03 SETUP -->

# &#x20; <div class="divider reveal">

# &#x20;   <div class="divider-num">03</div><div class="divider-line"></div>

# &#x20;   <div class="divider-label">Local Setup</div><div class="divider-line"></div>

# &#x20; </div>

# &#x20; <div class="reveal">

# &#x20;   <h2 class="section-title">Get Running <span class="accent">Locally</span></h2>

# &#x20;   <p style="color:var(--muted2);font-size:.9rem;margin-bottom:28px">Windows PowerShell instructions. Requires Python 3.10+ and Node.js.</p>

# &#x20;   <div class="card">

# &#x20;     <div class="steps">

# &#x20;       <div class="step">

# &#x20;         <div class="step-num"></div>

# &#x20;         <div class="step-body">

# &#x20;           <h4>Create \&amp; Activate Virtual Environment</h4>

# &#x20;           <pre>python -m venv .venv

# .\\.venv\\Scripts\\Activate.ps1</pre>

# &#x20;         </div>

# &#x20;       </div>

# &#x20;       <div class="step">

# &#x20;         <div class="step-num"></div>

# &#x20;         <div class="step-body">

# &#x20;           <h4>Install Python Dependencies</h4>

# &#x20;           <pre>pip install --upgrade pip

# pip install -r backend/requirements.txt</pre>

# &#x20;         </div>

# &#x20;       </div>

# &#x20;       <div class="step">

# &#x20;         <div class="step-num"></div>

# &#x20;         <div class="step-body">

# &#x20;           <h4>Generate Model Weights \&nbsp;<span style="color:var(--red);font-size:.78rem;font-family:'JetBrains Mono',monospace">CRITICAL — DO NOT SKIP</span></h4>

# &#x20;           <p>Downloads MNIST, applies color transforms, trains both models, saves <code>.pth</code> weight files. Skipping causes <code>FileNotFoundError</code> on startup.</p>

# &#x20;           <pre>cd backend

# python train\_biased\_model.py

# python train\_unbiased\_model.py</pre>

# &#x20;         </div>

# &#x20;       </div>

# &#x20;       <div class="step">

# &#x20;         <div class="step-num"></div>

# &#x20;         <div class="step-body">

# &#x20;           <h4>Start the FastAPI Backend</h4>

# &#x20;           <pre>uvicorn main:app --reload --host 0.0.0.0 --port 8000</pre>

# &#x20;           <p>Swagger docs available at <code>http://localhost:8000/docs</code></p>

# &#x20;         </div>

# &#x20;       </div>

# &#x20;       <div class="step">

# &#x20;         <div class="step-num"></div>

# &#x20;         <div class="step-body">

# &#x20;           <h4>Run the Frontend <span style="color:var(--muted);font-size:.8rem;font-weight:400">(optional — live Vercel app works too)</span></h4>

# &#x20;           <pre>cd frontend

# npm install

# npm start</pre>

# &#x20;           <p>Runs at <code>http://localhost:3000</code></p>

# &#x20;         </div>

# &#x20;       </div>

# &#x20;     </div>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <div class="reveal" style="margin-top:32px">

# &#x20;   <h3 style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:.04em;margin-bottom:20px">Environment <span style="color:var(--cyan)">Variables</span></h3>

# &#x20;   <div class="card" style="margin-bottom:16px">

# &#x20;     <div class="card-title" style="margin-bottom:12px">backend/.env</div>

# &#x20;     <pre><span class="cm"># Cloudinary — Object Storage for Grad-CAM images</span>

# CLOUDINARY\_CLOUD\_NAME=your\_cloud\_name

# CLOUDINARY\_API\_KEY=your\_api\_key

# CLOUDINARY\_API\_SECRET=your\_api\_secret

# 

# <span class="cm"># Redis — In-Memory Caching</span>

# REDIS\_URL=redis://your\_redis\_url:port

# 

# <span class="cm"># Weights \& Biases — MLOps Tracking</span>

# WANDB\_API\_KEY=your\_wandb\_api\_key</pre>

# &#x20;   </div>

# &#x20;   <div class="card">

# &#x20;     <div class="card-title" style="margin-bottom:12px">frontend/.env</div>

# &#x20;     <pre><span class="cm"># Local development</span>

# REACT\_APP\_API\_URL=http://localhost:8000</pre>

# &#x20;     <div class="alert warn">

# &#x20;       <span class="alert-icon">⚠️</span>

# &#x20;       <span><strong>Never hardcode the backend URL.</strong> In Vercel's dashboard, <code>REACT\_APP\_API\_URL</code> points to the Render production URL. Hardcoding breaks local dev and is a security risk.</span>

# &#x20;     </div>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <!-- 04 BACKEND -->

# &#x20; <div class="divider reveal">

# &#x20;   <div class="divider-num">04</div><div class="divider-line"></div>

# &#x20;   <div class="divider-label">Backend Architecture</div><div class="divider-line"></div>

# &#x20; </div>

# &#x20; <div class="reveal">

# &#x20;   <h2 class="section-title">Data <span class="accent">Architecture</span></h2>

# &#x20;   <div class="card">

# &#x20;     <p class="card-body" style="margin-bottom:20px">A hybrid storage strategy separates structured telemetry from heavy unstructured blobs, preventing database bloat and enabling fast CDN delivery to the frontend.</p>

# &#x20;     <div class="tbl-wrap">

# &#x20;       <table>

# &#x20;         <thead><tr><th>Store</th><th>Technology</th><th>What It Holds</th></tr></thead>

# &#x20;         <tbody>

# &#x20;           <tr><td><span class="pill pill-cyan">SQLite</span></td><td><code>predictions.db</code> · SQLAlchemy ORM</td><td>Timestamps, model choice, confidence scores, predicted class, Cloudinary URL</td></tr>

# &#x20;           <tr><td><span class="pill pill-green">Cloudinary</span></td><td>CDN Object Storage (S3-equivalent)</td><td>Raw Base64 Grad-CAM heatmap images</td></tr>

# &#x20;           <tr><td><span class="pill pill-orange">Redis</span></td><td>In-memory Cache</td><td>Hot prediction results — reduces redundant ML inference</td></tr>

# &#x20;         </tbody>

# &#x20;       </table>

# &#x20;     </div>

# &#x20;     <div class="alert info">

# &#x20;       <span class="alert-icon">💡</span>

# &#x20;       <span>The database stores only the Cloudinary CDN URL — not the raw Base64 blob. React fetches heatmaps directly from the CDN edge, never burdening the inference server.</span>

# &#x20;     </div>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <div class="reveal" style="margin-top:28px">

# &#x20;   <h3 style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:.04em;margin-bottom:20px">Model <span style="color:var(--cyan)">Retraining</span></h3>

# &#x20;   <div class="card">

# &#x20;     <p class="card-body" style="margin-bottom:16px">Retrain either model from scratch. Telemetry streams automatically to your W\&B dashboard during the run.</p>

# &#x20;     <pre><span class="cm"># Train the Biased Model — learns background color shortcuts</span>

# python train\_biased\_model.py

# 

# <span class="cm"># Train the Unbiased Model — learns geometric shapes</span>

# python train\_unbiased\_model.py</pre>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <!-- 05 SHORTCUT TRAP -->

# &#x20; <div class="divider reveal">

# &#x20;   <div class="divider-num">05</div><div class="divider-line"></div>

# &#x20;   <div class="divider-label">Testing the Shortcut Trap</div><div class="divider-line"></div>

# &#x20; </div>

# &#x20; <div class="reveal">

# &#x20;   <h2 class="section-title">See the <span class="accent">Cheat Live</span></h2>

# &#x20;   <p style="color:var(--muted2);font-size:.9rem;margin-bottom:24px">Follow these steps to trigger a live misclassification and watch Grad-CAM reveal what the biased model is actually "looking at."</p>

# &#x20;   <div class="card">

# &#x20;     <div class="steps">

# &#x20;       <div class="step">

# &#x20;         <div class="step-num"></div>

# &#x20;         <div class="step-body"><h4>Open an image editor (e.g. MS Paint)</h4><p>Create a square canvas — 500 × 500 px.</p></div>

# &#x20;       </div>

# &#x20;       <div class="step">

# &#x20;         <div class="step-num"></div>

# &#x20;         <div class="step-body"><h4>Flood-fill the background with <span style="color:var(--green)">Solid Green</span> or <span style="color:var(--red)">Solid Red</span></h4><p>Use the paint-bucket tool on the entire canvas.</p></div>

# &#x20;       </div>

# &#x20;       <div class="step">

# &#x20;         <div class="step-num"></div>

# &#x20;         <div class="step-body"><h4>Draw the digit <span style="color:var(--cyan)">"1"</span> in Solid White, centered</h4><p>Use a thick brush in the middle of the canvas.</p></div>

# &#x20;       </div>

# &#x20;       <div class="step">

# &#x20;         <div class="step-num"></div>

# &#x20;         <div class="step-body">

# &#x20;           <h4>Upload and compare both models side by side</h4>

# &#x20;           <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px">

# &#x20;             <div class="alert danger" style="margin:0;font-size:.82rem"><span class="alert-icon">🔴</span><span><strong>Biased Model</strong> misclassifies. Grad-CAM highlights background edges — the digit is ignored.</span></div>

# &#x20;             <div class="alert success" style="margin:0;font-size:.82rem"><span class="alert-icon">🟢</span><span><strong>Unbiased Model</strong> correctly identifies "1". Grad-CAM cleanly highlights the stroke.</span></div>

# &#x20;           </div>

# &#x20;         </div>

# &#x20;       </div>

# &#x20;     </div>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <!-- 06 CI/CD -->

# &#x20; <div class="divider reveal">

# &#x20;   <div class="divider-num">06</div><div class="divider-line"></div>

# &#x20;   <div class="divider-label">CI / CD Pipeline</div><div class="divider-line"></div>

# &#x20; </div>

# &#x20; <div class="reveal">

# &#x20;   <h2 class="section-title">Automated <span class="accent">Deployment</span></h2>

# &#x20;   <p style="color:var(--muted2);font-size:.9rem;margin-bottom:24px">Every <code>push</code> or <code>pull\_request</code> to <code>main</code> triggers this strict multi-stage GitHub Actions pipeline.</p>

# &#x20;   <div class="pipe-stage"><div class="pipe-num">01</div><div class="pipe-content"><h4>Environment Provisioning</h4><p>Spins up <code>ubuntu-latest</code> runner · Configures Python 3.10 for a clean, reproducible build.</p></div></div>

# &#x20;   <div class="pipe-stage"><div class="pipe-num">02</div><div class="pipe-content"><h4>Dependency Installation</h4><p>Installs all packages from <code>backend/requirements.txt</code> including PyTorch, Torchvision, and OpenCV.</p></div></div>

# &#x20;   <div class="pipe-stage"><div class="pipe-num">03</div><div class="pipe-content"><h4>Automated Tests — PyTest</h4><p>Mounts FastAPI via <code>TestClient</code> + HTTPX. Validates routing, endpoint health, and model weight integrity. <strong style="color:var(--red)">A single failure halts the pipeline.</strong></p><pre style="margin-top:12px">- name: Run PyTest Automated Tests

# &#x20; working-directory: ./backend

# &#x20; run: |

# &#x20;   pip install pytest httpx

# &#x20;   pytest test\_main.py -v</pre></div></div>

# &#x20;   <div class="pipe-stage" style="border-color:rgba(0,229,160,.2)"><div class="pipe-num" style="color:rgba(0,229,160,.3)">04</div><div class="pipe-content"><h4>Continuous Deployment ✓</h4><p>100% pass rate → signals Vercel and Render to pull and deploy the latest build globally.</p></div></div>

# 

# &#x20;   <div class="tbl-wrap" style="margin-top:24px">

# &#x20;     <table>

# &#x20;       <thead><tr><th>Component</th><th>Technology</th><th>Host</th><th>Trigger</th></tr></thead>

# &#x20;       <tbody>

# &#x20;         <tr><td><strong>Frontend UI</strong></td><td>React.js</td><td><span class="pill pill-cyan">Vercel</span></td><td>Auto on <code>main</code> merge</td></tr>

# &#x20;         <tr><td><strong>Backend API</strong></td><td>FastAPI · Python</td><td><span class="pill pill-green">Render</span></td><td>Auto on <code>main</code> merge</td></tr>

# &#x20;       </tbody>

# &#x20;     </table>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <!-- 07 TESTING -->

# &#x20; <div class="divider reveal">

# &#x20;   <div class="divider-num">07</div><div class="divider-line"></div>

# &#x20;   <div class="divider-label">Test Coverage</div><div class="divider-line"></div>

# &#x20; </div>

# &#x20; <div class="reveal">

# &#x20;   <h2 class="section-title">What Gets <span class="accent">Tested</span></h2>

# &#x20;   <div class="card" style="margin-bottom:24px">

# &#x20;     <p class="card-body" style="margin-bottom:16px">PyTest suite uses <code>TestClient</code> (FastAPI) + HTTPX to simulate full API cycles without a live server running.</p>

# &#x20;     <pre>pytest test\_main.py -v</pre>

# &#x20;   </div>

# &#x20;   <div class="test-grid">

# &#x20;     <div class="test-card"><div class="test-check">✓</div><div><h4>Endpoint Health</h4><p>Verifies <code>/docs</code> (Swagger UI) and all core API routes return <code>200 OK</code>.</p></div></div>

# &#x20;     <div class="test-card"><div class="test-check">✓</div><div><h4>Error Handling</h4><p>Confirms invalid routes correctly return <code>404 Not Found</code> fallback responses.</p></div></div>

# &#x20;     <div class="test-card"><div class="test-check">✓</div><div><h4>Model Loading</h4><p>Confirms both <code>.pth</code> weight files are accessible, loadable, and uncorrupted.</p></div></div>

# &#x20;   </div>

# &#x20;   <div class="alert warn" style="margin-top:16px">

# &#x20;     <span class="alert-icon">⚠️</span>

# &#x20;     <span>A <strong>100% pass rate</strong> is mandatory. Any single failing test immediately blocks the deployment pipeline.</span>

# &#x20;   </div>

# &#x20; </div>

# 

# &#x20; <!-- 08 TROUBLESHOOTING -->

# &#x20; <div class="divider reveal">

# &#x20;   <div class="divider-num">08</div><div class="divider-line"></div>

# &#x20;   <div class="divider-label">Troubleshooting</div><div class="divider-line"></div>

# &#x20; </div>

# &#x20; <div class="reveal">

# &#x20;   <h2 class="section-title">Common <span class="accent">Errors</span></h2>

# &#x20;   <div class="trouble"><div class="trouble-q"><span class="trouble-tag">FileNotFoundError</span> Server fails to start</div><div class="trouble-a">You skipped Step 3. Run both training scripts to generate the <code>.pth</code> weight files before launching FastAPI.</div></div>

# &#x20;   <div class="trouble"><div class="trouble-q"><span class="trouble-tag">ERR\_CONNECTION\_REFUSED</span> Image upload fails</div><div class="trouble-a">FastAPI isn't running or isn't reachable. Ensure the server is on port 8000 and <code>REACT\_APP\_API\_URL</code> in your frontend <code>.env</code> is <code>http://localhost:8000</code>.</div></div>

# &#x20;   <div class="trouble"><div class="trouble-q"><span class="trouble-tag">CUDA Error</span> Model fails to load</div><div class="trouble-a">No GPU needed. Weights load on CPU by default via <code>map\_location=torch.device('cpu')</code> in <code>main.py</code>.</div></div>

# &#x20; </div>

# 

# &#x20; <!-- FOOTER -->

# &#x20; <footer>

# &#x20;   <div class="footer-brand">Shortcut<span> Learning</span><br>Detector</div>

# &#x20;   <div class="footer-info">Project 45 · Team 85<br>GLA University · 4th Semester AIML<br>Mentor: Mr. Preshit Desai</div>

# &#x20; </footer>

# 

# </div>

# 

# <script>

# const cur=document.getElementById('cursor'),ring=document.getElementById('cursor-ring');

# let mx=0,my=0,rx=0,ry=0;

# document.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY;cur.style.left=mx+'px';cur.style.top=my+'px'});

# (function animRing(){rx+=(mx-rx)\*.12;ry+=(my-ry)\*.12;ring.style.left=rx+'px';ring.style.top=ry+'px';requestAnimationFrame(animRing)})();

# document.querySelectorAll('a,button,.card,.tech-chip,.pipe-stage,.trouble,.model-card').forEach(el=>{

# &#x20; el.addEventListener('mouseenter',()=>{cur.style.width='16px';cur.style.height='16px';ring.style.width='52px';ring.style.height='52px'});

# &#x20; el.addEventListener('mouseleave',()=>{cur.style.width='10px';cur.style.height='10px';ring.style.width='36px';ring.style.height='36px'});

# });

# const obs=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting)e.target.classList.add('visible')}),{threshold:.1,rootMargin:'0px 0px -40px 0px'});

# document.querySelectorAll('.reveal').forEach(el=>obs.observe(el));

# </script>

# </body>

# </html>

