// ===== app.js 안전가드 (중복 로드/리스너 방지) =====
(() => {
  if (window.__OCR_APP_INIT__) { console.log('[OCR] already initialized'); return; }
  window.__OCR_APP_INIT__ = true;

  // 전역 충돌 방지: 첫 로드에서만 세팅
  const API_BASE = window.__API_BASE__ || (window.__API_BASE__ = location.origin);

  // ---- 여기부터 네가 쓰던 코드 그대로 (이벤트 리스너 포함) ----
  const fileInput = document.getElementById('fileInput') || document.getElementById('file');
  const ocrBtn    = document.getElementById('ocrBtn')    || document.getElementById('run');
  const saveBtn   = document.getElementById('saveBtn')   || document.getElementById('save');
  const cvs       = document.getElementById('cvs')       || document.getElementById('canvas');
  const ctx       = cvs.getContext('2d');
  const statusEl  = document.getElementById('status');
  const jsonEl    = document.getElementById('json');

  function setStatus(m){ if(statusEl) statusEl.textContent = m; console.log('[STATUS]', m); }
  function showJSON(o){ if(jsonEl) jsonEl.textContent = JSON.stringify(o,null,2); }

  async function loadBitmapFromFile(file){
    if (!/^image\//.test(file.type)) throw new Error('이미지 파일만 업로드하세요.');
    try { return await createImageBitmap(file); }
    catch {
      return await new Promise((res, rej)=>{
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload  = ()=>{ URL.revokeObjectURL(url); res(img); };
        img.onerror = ()=>{ URL.revokeObjectURL(url); rej(new Error('이미지 로드 실패')); };
        img.src = url;
      });
    }
  }

  function prepareCanvas(ow, oh){
    const maxW = 940, scale = ow > maxW ? maxW/ow : 1;
    const vw = Math.round(ow*scale), vh = Math.round(oh*scale);
    const dpr = window.devicePixelRatio || 1;
    cvs.style.width = vw+'px'; cvs.style.height = vh+'px';
    cvs.width = Math.floor(vw*dpr); cvs.height = Math.floor(vh*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,cvs.width,cvs.height);
    return {vw, vh};
  }

  function mapQuad(quad, ow, oh, vw, vh){
    if (!Array.isArray(quad) || quad.length!==4) return null;
    const norm = quad.flat().every(v=>v>=0 && v<=1);
    return quad.map(([x,y]) => norm ? [x*vw, y*vh] : [x*vw/ow, y*vh/oh]);
  }
  function mapRect(box, ow, oh, vw, vh){
    if (!Array.isArray(box) || box.length!==4) return null;
    const norm = box.every(v=>v>=0 && v<=1);
    let x0,y0,w,h;
    if (norm){ x0=box[0]*vw; y0=box[1]*vh; w=(box[2]-box[0])*vw; h=(box[3]-box[1])*vh;
      if (w<=0||h<=0){ x0=box[0]*vw; y0=box[1]*vh; w=box[2]*vw; h=box[3]*vh; }
    } else { x0=box[0]*vw/ow; y0=box[1]*vh/oh; w=(box[2]-box[0])*vw/ow; h=(box[3]-box[1])*vh/oh;
      if (w<=0||h<=0){ x0=box[0]*vw/ow; y0=box[1]*vh/oh; w=box[2]*vw/ow; h=box[3]*vh/oh; }
    }
    return {x:x0,y:y0,w,h};
  }

  function visualize(ow, oh, vw, vh, items){
    ctx.save(); ctx.fillStyle='rgba(0,0,0,0.35)'; ctx.fillRect(0,0,vw,vh); ctx.restore();
    ctx.lineWidth=2; ctx.strokeStyle='yellow'; ctx.fillStyle='rgba(255,255,0,0.15)';
    let drawn=0;
    for (const it of (items||[])){
      if (it.quad){
        const q = mapQuad(it.quad, ow, oh, vw, vh);
        if (q){ ctx.beginPath(); ctx.moveTo(q[0][0],q[0][1]); for(let i=1;i<4;i++) ctx.lineTo(q[i][0],q[i][1]);
          ctx.closePath(); ctx.fill(); ctx.stroke(); drawn++; continue; }
      }
      const r = mapRect(it.bbox || it.box || it.rect, ow, oh, vw, vh);
      if (r){ ctx.fillRect(r.x,r.y,r.w,r.h); ctx.strokeRect(r.x,r.y,r.w,r.h); drawn++; }
    }
    return drawn;
  }

  // ----- OCR 버튼 -----
  runBtn.addEventListener('click', async ()=>{
    const file = fileEl?.files?.[0]; if(!file) return setStatus('파일이 없습니다.');
    try{
      setStatus('이미지 로딩 중…');
      const bmp = await loadBitmapFromFile(file);
      const ow=bmp.width, oh=bmp.height;
      const {vw, vh} = prepareCanvas(ow, oh);
      ctx.drawImage(bmp, 0, 0, vw, vh);

      setStatus('업로드 중…');
      const fd = new FormData(); fd.append('file', file);
      const res = await fetch(`${API_BASE}/api/ocr`, { method:'POST', body: fd, credentials:'include' });
      if (res.status === 401) return setStatus('로그인 필요(401)');
      if (!res.ok) return setStatus(`서버 오류: ${res.status} ${res.statusText}`);

      const j = await res.json(); if (jsonEl) jsonEl.textContent = JSON.stringify(j,null,2);
      const n = Array.isArray(j.items) ? j.items.length : 0;
      setStatus(`시각화 중… (items: ${n})`);
      let drawn = visualize(ow, oh, vw, vh, j.items || []);
      if (!drawn){ ctx.fillStyle='rgba(255,0,0,0.15)'; ctx.strokeStyle='red';
        const w=Math.round(vw*0.3), h=Math.round(vh*0.12);
        ctx.fillRect(10,10,w,h); ctx.strokeRect(10,10,w,h);
        setStatus('OCR 완료(검출 0개) · 디버깅 박스 표시'); return; }
      setStatus(`OCR 완료! (${drawn}개)`);
    }catch(err){ console.error(err); setStatus('클라이언트 오류: ' + (err?.message || err)); }
  });

  // ----- 캡처 저장(선택) -----
  if (saveBtn){
    saveBtn.addEventListener('click', ()=>{
      canvas.toBlob(async (blob)=>{
        if (!blob) return setStatus('캡처 실패');
        const fd = new FormData(); fd.append('file', blob, 'capture.png');
        try{
          const r = await fetch(`${API_BASE}/api/capture`, { method:'POST', body: fd, credentials:'include' });
          const j = await r.json().catch(()=>({ok:false}));
          setStatus(j.ok ? `캡처 저장: ${j.saved}` : '캡처 저장 실패');
        }catch(e){ setStatus('캡처 업로드 오류: ' + e.message); }
      });
    });
  }
})(); // <-- IIFE 끝
