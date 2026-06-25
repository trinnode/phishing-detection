import { useRef, useEffect } from 'react';

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawBarChart(canvas, data, options = {}) {
  try {
    const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const pad = { top: 20, right: 16, bottom: 36, left: 48 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);

  const maxVal = options.max || Math.max(...data.map(d => d.value), 0.01);
  const barGap = options.barGap || 6;
  const groupGap = options.groupGap || 20;
  const barCount = data.length;
  const groups = options.groups || 1;
  const barsPerGroup = barCount / groups;
  const groupWidth = chartW / groups;
  const barWidth = Math.min((groupWidth - groupGap - barGap * (barsPerGroup - 1)) / barsPerGroup, 40);

  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + chartH - (chartH * i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = '#5c6078';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText((maxVal * i / 4).toFixed(options.decimals || 2), pad.left - 8, y + 3);
  }

  data.forEach((d, i) => {
    const groupIdx = Math.floor(i / barsPerGroup);
    const inGroupIdx = i % barsPerGroup;
    const x = pad.left + groupIdx * groupWidth + groupGap / 2 + inGroupIdx * (barWidth + barGap);
    const barH = (d.value / maxVal) * chartH;
    const y = pad.top + chartH - barH;

    const grad = ctx.createLinearGradient(x, y, x, pad.top + chartH);
    grad.addColorStop(0, d.color || '#6c7ae0');
    grad.addColorStop(1, (d.color || '#6c7ae0') + '40');
    ctx.fillStyle = grad;

    const r = 3;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + barWidth - r, y);
    ctx.quadraticCurveTo(x + barWidth, y, x + barWidth, y + r);
    ctx.lineTo(x + barWidth, pad.top + chartH);
    ctx.lineTo(x, pad.top + chartH);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.fill();

    ctx.fillStyle = '#9094ab';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(d.label || '', x + barWidth / 2, h - pad.bottom + 16);

    ctx.fillStyle = '#e4e6ef';
    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillText(d.value.toFixed(options.decimals || 3), x + barWidth / 2, y - 4);
  });

  if (options.groupLabels) {
    ctx.fillStyle = '#5c6078';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    options.groupLabels.forEach((label, i) => {
      const cx = pad.left + i * groupWidth + groupWidth / 2;
      ctx.fillText(label, cx, h - 4);
    });
  }
  } catch(e) {}
}

function drawProgressChart(canvas, items, options = {}) {
  try {
    const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const rowH = Math.min((h - 8) / items.length, 28);
  const startY = (h - rowH * items.length) / 2;

  items.forEach((item, i) => {
    const y = startY + i * rowH;
    const pct = Math.min(item.value / (options.max || 1), 1);

    ctx.fillStyle = '#9094ab';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(item.label || '', 8, y + rowH / 2 + 3);

    ctx.fillStyle = 'rgba(255,255,255,0.04)';
    const barX = 80;
    const barW = w - barX - 52;
    const barH = 8;
    const barY = y + (rowH - barH) / 2;
    ctx.beginPath();
    roundRect(ctx, barX, barY, barW, barH, 4);
    ctx.fill();

    ctx.fillStyle = item.color || '#6c7ae0';
    ctx.beginPath();
    roundRect(ctx, barX, barY, Math.max(barW * pct, 4), barH, 4);
    ctx.fill();

    ctx.fillStyle = '#e4e6ef';
    ctx.font = '600 10px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(item.value.toFixed(options.decimals || 3), w - 4, y + rowH / 2 + 3);
  });
  } catch(e) {}
}

export function BarChart({ data, height = 200, max, decimals = 3, groups, groupLabels, barGap, groupGap }) {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) {
      drawBarChart(ref.current, data, { max, decimals, groups, groupLabels, barGap, groupGap });
    }
  }, [data, height, max, decimals, groups, groupLabels]);

  return (
    <canvas ref={ref} style={{ width: '100%', height }} />
  );
}

export function ProgressChart({ items, height = 120, max, decimals = 3 }) {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) {
      drawProgressChart(ref.current, items, { max, decimals });
    }
  }, [items, height, max, decimals]);

  return (
    <canvas ref={ref} style={{ width: '100%', height }} />
  );
}

export function SimpleBar({ value, max = 1, label, color, height = 6 }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div>
      {label && (
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>{label}</span>
          <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-primary)' }}>{typeof value === 'number' ? value.toFixed(3) : value}</span>
        </div>
      )}
      <div className="progressBar" style={{ height }}>
        <div className="progressFill" style={{ width: `${pct}%`, background: color || 'var(--accent)' }} />
      </div>
    </div>
  );
}
