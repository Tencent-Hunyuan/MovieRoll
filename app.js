(async () => {
const $=(s,root=document)=>root.querySelector(s);
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
let data;
const visibleVideos=new Set(),manualPauses=new WeakSet(),managedPauses=new WeakSet();
const activeVideos=new Set(),playRequests=new WeakSet(),releaseTimers=new Map(),savedTimes=new WeakMap();
const playButtons=new WeakMap();
let preferredVideo,playbackFrame;
function updatePlayButton(v){const button=playButtons.get(v);if(button)button.hidden=!v.paused;}
function pauseOffscreen(v){if(!v.paused){managedPauses.add(v);v.pause();}}
function releaseLater(v){
  if(releaseTimers.has(v)||!v.getAttribute('src'))return;
  releaseTimers.set(v,setTimeout(()=>{
    releaseTimers.delete(v);
    if(visibleVideos.has(v)||activeVideos.has(v))return;
    savedTimes.set(v,v.currentTime);pauseOffscreen(v);v.removeAttribute('src');v.load();
  },5000));
}
function loadVideo(v){
  clearTimeout(releaseTimers.get(v));releaseTimers.delete(v);
  if(!v.getAttribute('src')){v.src=v.dataset.video;v.load();}
}
function playVisible(v){
  if(document.hidden||manualPauses.has(v)||!activeVideos.has(v)||playRequests.has(v)||!v.paused)return;
  loadVideo(v);playRequests.add(v);
  v.play().then(()=>{
    v.parentElement.querySelector('.video-status')?.remove();
    if(document.hidden||!activeVideos.has(v))pauseOffscreen(v);
  }).catch(error=>{
    if(error.name!=='AbortError'&&activeVideos.has(v)&&!v.parentElement.querySelector('.video-status'))v.parentElement.append(el('span','video-status','Use the video controls to start playback.'));
  }).finally(()=>playRequests.delete(v));
}
function schedulePlayback(){
  if(playbackFrame)return;
  playbackFrame=requestAnimationFrame(()=>{
    playbackFrame=undefined;
    const candidates=[...visibleVideos].filter(v=>!manualPauses.has(v)).map(v=>{
      const r=v.getBoundingClientRect(),visible=Math.max(0,Math.min(r.bottom,innerHeight)-Math.max(r.top,0));
      return {v,top:Math.round(r.top),ratio:visible/r.height,distance:Math.abs((r.top+r.bottom)/2-innerHeight/2)};
    }).filter(x=>x.ratio>=.35).sort((a,b)=>Number(b.v===preferredVideo)-Number(a.v===preferredVideo)||b.ratio-a.ratio||a.distance-b.distance);
    const rows=new Map();
    for(const item of candidates){if(!rows.has(item.top))rows.set(item.top,[]);rows.get(item.top).push(item);}
    const focused=[...rows.values()].sort((a,b)=>Number(b.some(x=>x.v===preferredVideo))-Number(a.some(x=>x.v===preferredVideo))||Math.max(...b.map(x=>x.ratio))-Math.max(...a.map(x=>x.ratio))||Math.min(...a.map(x=>x.distance))-Math.min(...b.map(x=>x.distance))).slice(0,2).flat();
    const next=new Set(document.hidden?[]:focused.map(x=>x.v));
    for(const v of activeVideos)if(!next.has(v))pauseOffscreen(v);
    activeVideos.clear();for(const v of next)activeVideos.add(v);
    for(const v of next)playVisible(v);
    for(const v of visibleVideos)updatePlayButton(v);
  });
}
const videoObserver=new IntersectionObserver(entries=>{
  for(const {target:v,isIntersecting} of entries){
    if(isIntersecting){visibleVideos.add(v);clearTimeout(releaseTimers.get(v));releaseTimers.delete(v);}
    else{visibleVideos.delete(v);pauseOffscreen(v);releaseLater(v);}
  }
  schedulePlayback();
},{threshold:[0,.35,.75,1]});
window.addEventListener('scroll',schedulePlayback,{passive:true});
window.addEventListener('resize',schedulePlayback);
function media(record,alt,eager=false,withSound=false){
  if(record.video){
    const v=el('video');v.controls=true;v.muted=!withSound;v.defaultMuted=!withSound;v.volume=1;v.loop=true;v.playsInline=true;v.preload='none';if(record.poster)v.poster=record.poster;v.dataset.video=record.video;v.setAttribute('aria-label',alt);
    // Audible demo playback starts with a user gesture, not a blocked autoplay attempt.
    if(withSound)manualPauses.add(v);
    // Load on interaction so a poster-only tile still has working native controls.
    v.addEventListener('pointerdown',()=>{preferredVideo=v;loadVideo(v);});
    v.addEventListener('focus',()=>loadVideo(v));
    v.addEventListener('loadedmetadata',()=>{const time=savedTimes.get(v);if(time>0&&Number.isFinite(v.duration))v.currentTime=Math.min(time,v.duration-.05);savedTimes.delete(v);});
    v.addEventListener('pause',()=>{if(managedPauses.has(v))managedPauses.delete(v);else if(v.currentTime>0&&!v.ended){manualPauses.add(v);schedulePlayback();}});
    v.addEventListener('pause',()=>updatePlayButton(v));
    v.addEventListener('play',()=>{manualPauses.delete(v);updatePlayButton(v);if(!playRequests.has(v)){preferredVideo=v;schedulePlayback();}});
    queueMicrotask(()=>{
      const button=el('button','video-start','▶ Play');button.type='button';button.setAttribute('aria-label',`Play ${alt}`);
      button.addEventListener('click',()=>{preferredVideo=v;manualPauses.delete(v);loadVideo(v);if(withSound)v.play().catch(()=>updatePlayButton(v));else schedulePlayback();});
      playButtons.set(v,button);v.parentElement.append(button);updatePlayButton(v);
    });
    videoObserver.observe(v);return v;
  }
  if(record.poster){const image=el('img');image.src=record.poster;image.alt=alt;image.loading=eager?'eager':'lazy';image.decoding='async';image.width=864;image.height=480;return image;}
  const empty=el('div','empty-media');empty.innerHTML='<span class="placeholder-play" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="m9 6 9 6-9 6z"/></svg></span>';if(record.slot){const number=el('span','placeholder-number',String(record.slot).padStart(2,'0'));number.setAttribute('aria-hidden','true');empty.append(number);}empty.append(el('span','placeholder-label','Video coming soon'));return empty;
}
function promptDetails(text,label='Prompt'){
  const d=el('details','prompt-details'),s=el('summary','',label),icon=el('span');icon.setAttribute('aria-hidden','true');s.append(icon);d.append(s,el('p','',text));return d;
}
function renderAuthors(){
  $('#authors').replaceChildren(...data.authors.map(a=>{const n=el('span','',a.name),sup=el('sup','',a.affiliations.map(x=>x==='\\dag'?'†':x).join(','));n.append(sup);return n;}));
}
const icons={document:'<svg viewBox="0 0 24 24"><path d="M6 2h8l4 4v16H6zM14 2v5h4M9 12h6M9 16h6"/></svg>',arxiv:'<span style="font:italic 19px Georgia,serif">aχ</span>',code:'<svg viewBox="0 0 24 24"><path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-13-2 20"/></svg>',models:'<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M8 10h.01M16 10h.01M8 14c2 3 6 3 8 0" stroke-linecap="round"/></svg>',video:'<svg viewBox="0 0 24 24"><rect x="2" y="4" width="20" height="16" rx="3"/><path d="m10 8 6 4-6 4z"/></svg>'};
Object.assign(icons,{
 arxiv:'<img class="arxiv-mark" src="assets/logos/resource-arxiv.svg" alt="">',
 github:'<img src="assets/logos/resource-github.svg" alt="">',
 huggingface:'<img src="assets/logos/resource-huggingface.svg" alt="">'
});
function renderResources(){
  data.resources.forEach(r=>{const card=el(r.href?'a':'div',`resource-card${!r.href?' unavailable':''}`);if(r.href){card.href=r.href;if(!r.href.startsWith('#')){card.target='_blank';card.rel='noopener';}}else card.setAttribute('aria-label',`${r.label}: ${r.detail}`);const icon=el('span','resource-icon');icon.setAttribute('aria-hidden','true');icon.innerHTML=icons[r.icon]||'';const text=el('span');text.append(el('span','resource-label',r.label),el('span','resource-detail',r.detail));card.append(icon,text);$('#resource-cards').append(card);});
}
function renderDemo(){
  if(!data.demo?.video)return;
  const v=media(data.demo,data.demo.title||'MovieRoll demo',true,true);
  if(['localhost','127.0.0.1','[::1]'].includes(location.hostname)&&data.demo.video.startsWith('https://github.com/user-attachments/')){
    v.addEventListener('error',()=>{
      if(v.dataset.localFallback)return;
      v.dataset.localFallback='true';
      v.dataset.video='assets/videos/movieroll-demo.mp4';
      v.src=v.dataset.video;
      v.load();
      v.play().catch(()=>updatePlayButton(v));
    });
  }
  $('#demo-slot').replaceChildren(v);
}
function card(record){
  const c=el('article',`media-card${record.pending?' pending':''}`);c.dataset.mediaId=record.id;c.dataset.slot=record.slot;
  const box=el('div','media-box');box.append(media(record,`${record.title}, ${record.model}`));if(record.poster&&!record.video)box.append(el('span','media-badge','Image preview'));
  const caption=el('div','media-caption');caption.append(el('h4','',record.pending?`Case ${String(record.slot).padStart(2,'0')}`:record.title));caption.append(el('p','',record.pending?'Video pending':`${record.duration}s${record.eventPrompts?' · '+record.eventPrompts.length+' events':''}`));
  c.append(box,caption);if(record.prompt)c.append(promptDetails(record.prompt,record.eventPrompts?'Event prompts':'Generation prompt'));else c.append(el('p','pending-caption','Prompt to be added'));return c;
}
function renderShowcases(){
  $('#single-base').replaceChildren(...data.showcases.singleBase.map(card));
  const dist=$('#single-distilled');dist.replaceChildren();
  for(let i=0;i<Math.ceil(data.showcases.singleDistilled.length/12);i++){
    const records=data.showcases.singleDistilled.slice(i*12,i*12+12);
    const durations=[...new Set(records.map(record=>record.duration))];
    const label=durations.length===1?`${durations[0]} SECONDS · SET ${i+1}`:`SET ${i+1}`;
    const set=el('div','showcase-set');set.append(el('p','set-label',label));
    const grid=el('div','showcase-grid');grid.dataset.set=i+1;grid.append(...records.map(card));set.append(grid);dist.append(set);
  }
  $('#multi-base').replaceChildren(...data.showcases.multiBase.map(card));$('#multi-distilled').replaceChildren(...data.showcases.multiDistilled.map(card));
}
function renderComparisons(){
  data.comparisons.forEach((g,i)=>{const section=el('section','comparison-group');section.dataset.task=g.task;section.dataset.duration=g.duration;section.dataset.scene=g.id;
    const heading=el('div','comparison-heading');heading.append(el('h4','',`${g.duration}-Second Comparison`),el('p','',g.title));section.append(heading);
    const grid=el('div','comparison-wall');const outputs=[...g.outputs].sort((a,b)=>Number(a.method.startsWith('MovieRoll'))-Number(b.method.startsWith('MovieRoll')));
    const rowCount=Math.ceil(outputs.length/4), rowSize=Math.floor(outputs.length/rowCount), fullerRows=outputs.length%rowCount;
    const rowSizes=Array.from({length:rowCount},(_,r)=>rowSize+(r<fullerRows?1:0));let row=0,column=0;
    outputs.forEach(o=>{const c=el('article',`comparison-card${o.method.startsWith('MovieRoll')?' ours':''}`);c.dataset.method=o.method;c.style.setProperty('--desktop-span',12/rowSizes[row]);if(++column===rowSizes[row]){row++;column=0;}const box=el('div','media-box');box.append(media(o,`${o.method}, ${g.title}, ${g.duration} seconds`));if(!o.video)box.append(el('span','media-badge','Image preview'));c.append(el('h5','comparison-label',o.method),box);grid.append(c);});
    section.append(grid,promptDetails(g.prompt,g.eventPrompts?'Shared event prompts':'Shared generation prompt'));$(`#${g.task==='Single event'?'single':'multi'}-comparisons`).append(section);
  });
}
// Method order follows the current paper, then Infrastructure.
const methodSections=[
  {
    "title": "Overview",
    "image": "overview",
    "alt": "MovieRoll training and inference framework with WinTF, CuLT-DMD, HieraKV and rolling generation",
    "description": "MovieRoll is a base-to-distilled framework for long-form video generation. Window-wise Teacher Forcing adapts a pretrained bidirectional model into MovieRoll-Base. ODE trajectories from the Base model initialize a few-step student, and CuLT-DMD trains it on long self-generated rollouts. Both models jointly denoise a rolling window and use HieraKV with KV recomputation to manage historical context. AnnealCFG reduces guidance toward the clean endpoint in MovieRoll-Base."
  },
  {
    "title": "Window-wise Teacher Forcing (WinTF)",
    "image": "winTF",
    "alt": "Phase-offset training windows, progressive timestep sampling and the WinTF attention mask",
    "description": "WinTF trains multiple rolling windows in one pass. Within each window, later chunks receive progressively higher noise levels. Chunks attend bidirectionally within their window and causally to preceding ground-truth history, while attention between separate noisy windows is masked. Random phase offsets cover the growing window during prefill and the full window during decoding. Each window shares an event-specific prompt, enabling supervision across successive events."
  },
  {
    "title": "Hierarchical KV-Cache Compression across Layers (HieraKV)",
    "image": "hireakv",
    "alt": "Layer-wise historical attention mass, history-attention entropy and sink–recent coverage",
    "description": "Historical attention varies with network depth. Shallow blocks attend broadly to earlier chunks, while deeper blocks concentrate on the sink and recent history. HieraKV therefore retains full available history in shallow blocks and only the sink and most recent historical chunk in deeper blocks. Intermediate history remains accessible through the shallow-block outputs, and active-window attention stays bidirectional at every depth."
  },
  {
    "title": "Cumulative Long Tuning with DMD (CuLT-DMD)",
    "image": "cult_dmd",
    "alt": "Local DMD supervision and a comparison of per-window updates with rollout-level gradient accumulation",
    "description": "CuLT-DMD keeps model parameters fixed throughout a long self-rollout and averages local gradients before updating at the rollout boundary. Each supervision window is scored with a short prefix of detached generated history. A frozen bidirectional teacher and a trainable critic evaluate the same noisy score clip, and their prediction difference supplies a detached update signal for the current window. Each scaled local loss is backpropagated immediately and its graph released. The generator and critic thus aggregate supervision across temporal positions while retaining one local computation graph at a time."
  },
  {
    "id": "annealcfg",
    "title": "Mitigating Long-Horizon Degradation with CFG Annealing (AnnealCFG)",
    "image": "annealcfg_method_analysis",
    "alt": "Extrapolation impulse and late-stage impulse share for constant and annealed CFG across four stages of 30-second generation",
    "description": "Guidance-induced deviations can propagate when a fully denoised chunk becomes history. AnnealCFG keeps stronger guidance at high noise and reduces it toward the clean endpoint. The analysis below uses 32 paired prompts and matched seeds across 30s generation. It measures extrapolation impulse, the magnitude of the additional latent update contributed by CFG, and shows reduced late-stage extrapolation with annealed guidance.",
    "equations": [
      "<math display=\"block\" aria-label=\"Guided velocity equals unconditional velocity plus w of t times the conditional velocity difference\"><msub><mi>v</mi><mtext>cfg</mtext></msub><mo>=</mo><msub><mi>v</mi><mtext>uncond</mtext></msub><mo>+</mo><mi>w</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>[</mo><msub><mi>v</mi><mtext>cond</mtext></msub><mo>−</mo><msub><mi>v</mi><mtext>uncond</mtext></msub><mo>]</mo></math>",
      "<math display=\"block\" aria-label=\"w of t equals one plus w max minus one times t, for normalized t from zero to one\"><mi>w</mi><mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo><mn>1</mn><mo>+</mo><mo>(</mo><msub><mi>w</mi><mtext>max</mtext></msub><mo>−</mo><mn>1</mn><mo>)</mo><mi>t</mi><mo>,</mo><mspace width=\"1em\"/><mi>t</mi><mo>∈</mo><mo>[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>]</mo></math>"
    ],
    "note": "With normalized noise decreasing from t = 1 to t = 0, guidance falls from wₘₐₓ to 1 and recovers the conditional prediction at the clean endpoint. The main Base configuration uses 4 → 1. MovieRoll-Distilled uses no CFG at inference."
  },
  {
    "id": "cache-recomputation",
    "title": "Threshold-Triggered KV Recomputation for Extrapolation",
    "image": "recompute",
    "alt": "Full-history and sliding-window extrapolation compared with KV recomputation from selected clean chunks",
    "description": "When history reaches a threshold or the event prompt switches, MovieRoll selects the sink chunk, representative event-middle chunks, and recent chunks. A causal forward pass re-encodes their clean latents in temporal order at t = 0 without text conditioning. Generation resumes from this rebuilt cache. This bounds historical storage and removes the retained KV states’ dependencies on discarded chunks."
  },
  {
    "id": "prompt-rewriting-method",
    "title": "Streaming Video Prompt Rewriting Skill",
    "description": "Long-form generation benefits from sustained actions and consistent entities. The prompt-rewriting skill expands short instructions into ongoing actions and detailed scene descriptions. Single-event prompts develop one scene, while multi-event prompts use preceding rewritten prompts to maintain continuity."
  },
  {
    "title": "Flash Block-wise Causal Attention (FBCA)",
    "image": "FBCA",
    "alt": "Block-causal visibility, KV unfolding and grouping, and variable-length FlashAttention",
    "description": "FBCA implements WinTF’s attention structure by gathering the keys and values visible to each query block. The visible sequence contains the shared prefix, preceding historical states, and the current block. Variable-length FlashAttention evaluates the packed sequences independently. Consecutive blocks are grouped to bound the buffer required by each call while preserving visible context. In the paper’s H20 measurements, FBCA has the lowest forward and training times, and preparation remains below 0.1s across multiple input shapes.",
    "tables": [
      {
        "title": "Attention Efficiency on One H20 GPU",
        "caption": "Full-history attention with three or five windows on one NVIDIA H20 GPU, evaluated with fixed and varying input shapes. Lower values are better.",
        "headers": [
          "Windows",
          "Implementation",
          "Forward (ms)",
          "Forward + Backward (ms)",
          "Train Peak (GiB)",
          "Preparation (s)",
          "Multiple-Shape Total (s)",
          "Step Peak (GiB)"
        ],
        "rows": [
          [
            "3",
            "SDPA (cuDNN)",
            "1678.64",
            "6802.64",
            "64.19",
            "41.76",
            "249.83",
            "62.14"
          ],
          [
            "3",
            "FlexAttention",
            "699.81",
            "2789.95",
            "10.37",
            "57.48",
            "141.73",
            "28.61"
          ],
          [
            "3",
            "FBCA",
            "541.44",
            "1878.47",
            "13.35",
            "<0.10",
            "56.82",
            "13.19"
          ],
          [
            "5",
            "SDPA (cuDNN)",
            "—",
            "OOM",
            "—",
            "—",
            "OOM",
            "—"
          ],
          [
            "5",
            "FlexAttention",
            "1891.92",
            "7644.83",
            "18.50",
            "138.58",
            "369.94",
            "87.13"
          ],
          [
            "5",
            "FBCA",
            "1491.18",
            "5204.46",
            "31.51",
            "<0.10",
            "156.94",
            "31.12"
          ]
        ]
      }
    ]
  },
  {
    "title": "Parallel Streaming Variational Autoencoder (PS-VAE)",
    "image": "ps_vae",
    "alt": "Temporal decoder state across chunks and row-sharded parallel decoding with halo exchange",
    "description": "PS-VAE decodes latent chunks as they arrive, preserving causal-convolution inputs, temporal-upsampling state, and bottleneck-attention keys and values between chunks. For spatial parallelism, each device computes the input and middle blocks, then decodes a contiguous row band through the upsampling path. Neighboring devices exchange boundary rows, and an all-gather assembles the decoded frames. This preserves single-device streaming decoding under the same cache policy, up to numerical differences. For 1441-frame decoding, eight H20 GPUs achieve 3.97× speedup over single-GPU PS-VAE.",
    "tables": [
      {
        "title": "PS-VAE Decoding Efficiency and Numerical Agreement",
        "caption": "1441 frames at 480 × 864 on NVIDIA H20 GPUs with FP16. Speedup and agreement PSNR use single-GPU PS-VAE as the reference.",
        "headers": [
          "Method",
          "GPUs",
          "Time (s)",
          "Throughput (FPS)",
          "Speedup",
          "Memory (GiB)",
          "Reconstruction PSNR (dB)",
          "Agreement PSNR (dB)"
        ],
        "rows": [
          [
            "VAE (Full Sequence)",
            "1",
            "OOM",
            "—",
            "—",
            "—",
            "—",
            "—"
          ],
          [
            "Temporal Tiling (32 frames)",
            "1",
            "219.61",
            "6.56",
            "0.80×",
            "37.11",
            "38.43",
            "42.82"
          ],
          [
            "Spatial Tiling (64 px)",
            "1",
            "273.90",
            "5.26",
            "0.64×",
            "20.67",
            "38.69",
            "43.20"
          ],
          [
            "Temporal + Spatial Tiling (64 frames, 256 px)",
            "1",
            "355.11",
            "4.06",
            "0.50×",
            "16.15",
            "39.91",
            "49.15"
          ],
          [
            "PS-VAE",
            "1",
            "176.15",
            "8.18",
            "1.00×",
            "28.84",
            "40.46",
            "Ref."
          ],
          [
            "PS-VAE",
            "2",
            "101.96",
            "14.13",
            "1.73×",
            "18.46",
            "40.46",
            "86.21"
          ],
          [
            "PS-VAE",
            "4",
            "64.79",
            "22.24",
            "2.72×",
            "15.51",
            "40.46",
            "86.22"
          ],
          [
            "PS-VAE",
            "8",
            "44.37",
            "32.47",
            "3.97×",
            "14.83",
            "40.46",
            "86.22"
          ]
        ]
      }
    ]
  }
];
const userStudySections=[
 {title:'Single-Event Preferences',image:'gsb_single_event',alt:'Win, tie, and loss rates for MovieRoll-Distilled against ten single-event baselines',description:'Responses are pooled across 10-, 30-, and 60-second videos. Ties remain in the denominator of all rates. MovieRoll-Distilled receives a majority of preferences against every compared baseline.'},
 {title:'Multi-Event Preferences',image:'gsb_multi_event',alt:'Win, tie, and loss rates for MovieRoll-Distilled against five multi-event baselines',description:'Responses are pooled across 30-second videos with three events and 60-second videos with six events. Ties remain in the denominator of all rates. MovieRoll-Distilled receives a majority of preferences against every compared baseline.'}
];
const figureDialog=$('#figure-lightbox'),figureViewport=$('#lightbox-viewport'),enlargedImage=$('#lightbox-image'),zoomFigure=$('#zoom-figure');
let figureTrigger;
function figureUrl(name){
  const version=data.figureVersions?.[name];
  return `assets/figures/${name}.webp${version?`?v=${encodeURIComponent(version)}`:''}`;
}
function openFigure(method,trigger){
  figureTrigger=trigger;
  figureDialog.classList.remove('is-zoomed');
  $('#lightbox-title').textContent=method.title;
  enlargedImage.src=figureUrl(method.image);
  enlargedImage.alt=method.alt;
  const [width,height]=data.figureSizes[method.image];
  enlargedImage.width=width;enlargedImage.height=height;
  figureDialog.style.setProperty('--zoom-width',`${Math.max(width,window.innerWidth*1.6)}px`);
  zoomFigure.setAttribute('aria-pressed','false');
  zoomFigure.setAttribute('aria-label','Zoom into figure');
  $('#lightbox-help').textContent='Click or tap the image to zoom · Esc to close';
  document.documentElement.classList.add('figure-open');
  figureDialog.showModal();
  figureViewport.scrollTo(0,0);
  $('#close-figure').focus({preventScroll:true});
}
function restoreFigurePage(){
  if(figureDialog.open)return;
  document.documentElement.classList.remove('figure-open');
  figureDialog.classList.remove('is-zoomed');
  figureTrigger?.focus({preventScroll:true});
}
function closeFigure(){figureDialog.close();restoreFigurePage();}
$('#close-figure').addEventListener('click',closeFigure);
figureDialog.addEventListener('cancel',event=>{event.preventDefault();closeFigure();});
figureDialog.addEventListener('close',restoreFigurePage);
figureDialog.addEventListener('click',event=>{
  if(event.target!==figureDialog)return;
  const r=figureDialog.getBoundingClientRect();
  if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)closeFigure();
});
zoomFigure.addEventListener('click',()=>{
  const zoomed=figureDialog.classList.toggle('is-zoomed');
  zoomFigure.setAttribute('aria-pressed',String(zoomed));
  zoomFigure.setAttribute('aria-label',zoomed?'Fit figure to screen':'Zoom into figure');
  $('#lightbox-help').textContent=zoomed?'Scroll to inspect · Click image to fit · Esc to close':'Click or tap the image to zoom · Esc to close';
  requestAnimationFrame(()=>{
    if(zoomed)figureViewport.scrollTo((figureViewport.scrollWidth-figureViewport.clientWidth)/2,(figureViewport.scrollHeight-figureViewport.clientHeight)/2);
    else figureViewport.scrollTo(0,0);
  });
});
function renderFigureSections(sections,container,headingTag='h3'){
 sections.forEach(m=>{
  const article=el('article','method-block');if(m.id)article.id=m.id;
  article.append(el(headingTag,'',m.title),el('p','method-description',m.description));
  if(m.image){
   article.dataset.figure=m.image;
   const figure=el('figure','method-figure'),img=el('img');
   img.src=figureUrl(m.image);img.alt=m.alt;img.loading='lazy';img.decoding='async';
   [img.width,img.height]=data.figureSizes[m.image];
   const link=el('button','figure-link');link.type='button';link.setAttribute('aria-haspopup','dialog');
   link.setAttribute('aria-controls','figure-lightbox');link.setAttribute('aria-label',`Enlarge figure: ${m.title}`);
   link.addEventListener('click',()=>openFigure(m,link));
   link.append(img,el('span','figure-hint','Enlarge figure ⤢'));figure.append(link);article.append(figure);
  }
  if(m.equations){const equations=el('div','method-equations');m.equations.forEach(markup=>{const row=el('div','equation-row');row.innerHTML=markup;equations.append(row);});article.append(equations);}
  if(m.note)article.append(el('p','method-description equation-note',m.note));
  if(m.tables)m.tables.forEach(spec=>{
    const region=el('div','method-evidence');region.append(el('h4','',spec.title),el('p','section-note',spec.caption));
    const scroller=el('div','table-scroller');scroller.tabIndex=0;scroller.setAttribute('role','region');scroller.setAttribute('aria-label',spec.title+'; scroll horizontally for all columns');
    const table=el('table','evidence-table'),head=el('thead'),header=el('tr');
    spec.headers.forEach(label=>{const cell=el('th','',label);cell.scope='col';header.append(cell);});head.append(header);
    const body=el('tbody');spec.rows.forEach(values=>{const row=el('tr',values.includes('FBCA')||values[0]==='PS-VAE'?'our-row':'');values.forEach(value=>row.append(el('td','',value)));body.append(row);});
    table.append(head,body);scroller.append(table);region.append(scroller);article.append(region);
  });

  $(container).append(article);
 });
}
function renderResults(){
 const [task,duration]=$('#results-setting').value.split(':'),rows=data.results[task][duration],names=['Aesthetic','Dynamics','Smoothness','Semantic','Naturalness','Drift: aesthetic','Drift: smoothness','Drift: semantic','Drift: naturalness','Total'],order=[9,4,1,0,2,3,5,6,7,8];
 const head=el('thead'),tr=el('tr');tr.append(el('th','','Method'));order.forEach(i=>tr.append(el('th','',names[i]+'\u00a0↑')));head.append(tr);const body=el('tbody'),best=names.map((_,i)=>Math.max(...rows.map(r=>r.values[i])));
 rows.forEach(r=>{const row=el('tr',r.method.startsWith('MovieRoll')?'our-row':''),cell=el('td');if(r.method.startsWith('MovieRoll'))cell.append(el('span','our-dot'));cell.append(document.createTextNode(r.method));row.append(cell);order.forEach(i=>row.append(el('td',r.values[i]===best[i]?'best':'',i===9?r.values[i].toFixed(3):String(r.values[i]))));body.append(row);});$('#results-table').replaceChildren(head,body);
}
function renderCitation(){
 $('#citation-code').textContent=data.citation;$('#copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(data.citation);$('#copy-citation').textContent='Copied';$('#citation-status').textContent='Citation copied.';}catch{const range=document.createRange();range.selectNodeContents($('#citation-code'));const s=window.getSelection();s.removeAllRanges();s.addRange(range);$('#citation-status').textContent='Copy the selected citation.';}});
}
$('#results-setting').addEventListener('change',renderResults);document.addEventListener('visibilitychange',()=>{if(document.hidden)activeVideos.forEach(pauseOffscreen);schedulePlayback();});
try{data=await window.loadMovieRollContent();renderAuthors();renderResources();renderDemo();renderShowcases();renderComparisons();renderFigureSections(methodSections,'#method-figures');renderFigureSections(userStudySections,'#user-study-figures','h4');renderResults();renderCitation();}catch(e){$('#load-error').hidden=false;console.error(e);}
})();
