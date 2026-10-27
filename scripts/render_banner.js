// Renders docs/assets/banner.html to a static, SMIL-animated SVG (README banner).
// usage: node scripts/render_banner.js docs/assets/banner.svg
let _s=7; const rand=()=>{_s=(_s*16807)%2147483647; return (_s-1)/2147483646;};

class El{constructor(t){this.tag=t;this.attrs={};this.children=[];this.style={};}
 setAttribute(k,v){this.attrs[k]=v} appendChild(c){this.children.push(c);return c}
 replaceChildren(){this.children=[]}
 ser(){let a=Object.entries(this.attrs).map(([k,v])=>`${k}="${v}"`).join(" ");
  if(this.style.display==="none") a+=' display="none"';
  if(this.id) a+=` id="${this.id}"`;
  return `<${this.tag}${a?" "+a:""}>`+this.children.map(c=>c.ser()).join("")+`</${this.tag}>`}}
const root=new El("svg");
global.document={getElementById:()=>root,createElementNS:(ns,t)=>new El(t)};
global.window={matchMedia:()=>({matches:true})};

const W = 72, H = 48, S = 10, NS = "http://www.w3.org/2000/svg";
const P = {
  bg: "#191433", scan: "#211a43",
  body: "#ffd45c", shade: "#f2b33d", spot: "#191433", fin: "#ffe9a8", ray: "#f7cf6e",
  eye: "#191433", led: "#5cf2ff", blush: "#ff6fb1",
  shaft: "#7d74b0", shaftHi: "#a59de0", collar: "#3b8fa6",
  hub: "#5cf2ff", hubShade: "#3b8fa6", bolt: "#ffffff",
  cone: "#ff6fb1", coneHi: "#ffb3d6",
  bladeHi: "#ffb3d6", blade: "#ff6fb1", bladeLo: "#c94a8c", bladeBack: "#9a3c78", cuff: "#5e3570", tip: "#5cf2ff",
  blur1: "#b44c8f", blur2: "#5e3570", blur3: "#3a2a5e",
  frame: "#3d3470",
  wake: ["#5cf2ff", "#3b8fa6", "#ffd45c"], wakeFar: ["#2d4f73", "#4a2f62"],
};

const svg = document.getElementById("art");
const rect = (parent, x, y, color, w = 1, h = 1) => {
  const r = document.createElementNS(NS, "rect");
  r.setAttribute("x", x * S); r.setAttribute("y", y * S);
  r.setAttribute("width", w * S); r.setAttribute("height", h * S);
  r.setAttribute("fill", color); parent.appendChild(r); return r;
};
const layer = () => { const g = document.createElementNS(NS, "g"); svg.appendChild(g); return g; };

// background + scanlines
const bgL = layer();
rect(bgL, 0, 0, P.bg, W, H);
for (let y = 0; y < H; y += 2) rect(bgL, 0, y, P.scan, W, 0).setAttribute("height", 3);

const wakeL = layer(), baseL = layer(), bladeL = layer();

// ---- koi ----
const grid = new Map();
const put = (x, y, c) => { if (x >= 0 && x < W && y >= 0 && y < H) grid.set(`${x},${y}`, c); };
const get = (x, y) => grid.get(`${x},${y}`);
const cx = 26, cy = 24, rx = 21, ry = 10;
const ryAt = x => { const t = (x - cx) / rx; return ry * (1 - 0.78 * Math.max(0, t)) * (1 - 0.08 * Math.max(0, -t)); };
const inBody = (x, y) => ((x - cx) / rx) ** 2 + ((y - cy) / ryAt(x)) ** 2 <= 1;
const topOf = x => cy - ryAt(x) * Math.sqrt(Math.max(0, 1 - ((x - cx) / rx) ** 2));

for (let x = 23; x < 37; x++) {
  const h = x <= 26 ? Math.max(0, Math.round(5 - (26 - x) * 0.8)) : Math.max(0, Math.round(5 - (x - 26) * 0.45));
  const t = Math.floor(topOf(x));
  for (let y = t - h; y <= t; y++) put(x, y, x % 2 ? P.ray : P.fin);
}
for (let x = 0; x < W; x++) for (let y = 0; y < H; y++)
  if (((x - 17) / 4.5) ** 2 + ((y - 33) / 2.4) ** 2 <= 1 && x <= 20) put(x, y, (x - y) % 3 === 0 ? P.ray : P.fin);
for (let x = 0; x < W; x++) for (let y = 0; y < H; y++)
  if (inBody(x, y)) put(x, y, (y - cy) > ryAt(x) * 0.4 ? P.shade : P.body);
grid.delete(`${cx},${cy + ry}`);

const skin = [P.body, P.shade, P.fin, P.ray];
[[25, 19, 2], [33, 24, 2], [20, 28, 1], [28, 29, 1]].forEach(([sx, sy, r]) => {
  for (let x = sx - r; x <= sx + r; x++) for (let y = sy - r; y <= sy + r; y++) {
    const corner = r >= 2 && Math.abs(x - sx) === r && Math.abs(y - sy) === r;
    if (!corner && skin.includes(get(x, y))) put(x, y, P.spot);
  }
});
[[12, 22], [13, 22], [12, 23], [13, 23]].forEach(([x, y]) => put(x, y, P.eye));
put(13, 22, P.led);
put(14, 25, P.blush); put(15, 25, P.blush);

// ---- drive train: shaft, collar, hub, spinner ----
put(47, cy, P.shaft); put(48, cy, P.shaft); put(48, cy - 1, P.shaftHi);
put(46, cy - 1, P.collar); put(46, cy + 1, P.collar);
// hub (rounded 2x5 with shading and a bolt)
for (let y = cy - 2; y <= cy + 2; y++) {
  if (Math.abs(y - cy) < 2) put(49, y, P.hubShade);
  put(50, y, P.hub);
}
put(50, cy, P.bolt);
// spinner cone
put(51, cy - 1, P.cone); put(51, cy, P.coneHi); put(51, cy + 1, P.cone); put(52, cy, P.cone);

// frame corners
[[2, 2, 1, 1], [W - 3, 2, -1, 1], [2, H - 3, 1, -1], [W - 3, H - 3, -1, -1]].forEach(([x0, y0, sx, sy]) => {
  for (let i = 0; i < 4; i++) { put(x0 + sx * i, y0, P.frame); put(x0, y0 + sy * i, P.frame); }
});
grid.forEach((c, k) => { const [x, y] = k.split(",").map(Number); rect(baseL, x, y, c); });

// ---- detailed two-blade propeller ----
// Side view of a rotating prop: apparent blade length shrinks to edge-on and grows back,
// showing the dark back face on the return half of the cycle.
// [apparent length, face: "front" | "back", blur level]
const FRAMES = [
  [11, "front", 0], [10, "front", 1], [7, "front", 2], [3, "edge", 3],
  [7, "back", 2], [10, "back", 1],
];
const frames = FRAMES.map(([len, face, blur], i) => {
  const g = document.createElementNS(NS, "g");
  g.style.display = i === 0 ? "" : "none";

  // curved motion blur trailing behind the blade (drawn first, under the blade)
  if (blur) {
    const span = 11;
    for (let d = 3; d <= span; d++) {
      const t = d / span;
      const bx = 53 + Math.round(t * t * 2);
      const c = blur >= 2 ? (t > 0.75 ? P.blur3 : P.blur2) : (t > 0.75 ? P.blur2 : P.blur1);
      rect(g, bx, cy - d, c); rect(g, bx, cy + d, c);
      if (blur >= 2 && d > 4) { rect(g, bx + 1, cy - d, P.blur3); rect(g, bx + 1, cy + d, P.blur3); }
    }
  }

  const blade = dir => {
    for (let d = 3; d <= len + 2; d++) {
      const y = cy + dir * d;
      const frac = (d - 3) / Math.max(1, len - 1);
      if (d === 3) { rect(g, 49, y, P.cuff, 3, 1); continue; }      // root cuff
      if (face === "edge") { rect(g, 50, y, P.bladeLo); continue; }
      // paddle-shaped chord: narrow at the root, widest around 40%, tapering to the tip
      const w = len >= 6 ? Math.max(1, Math.round(1 + 2.4 * Math.sin(Math.PI * Math.min(1, frac * 0.85 + 0.12)))) : 2;
      const x0 = 50 - Math.floor((w - 1) / 2);
      for (let i = 0; i < w; i++) {
        let c;
        if (face === "front") c = i === 0 ? P.bladeHi : (i === w - 1 && w > 1 ? P.bladeLo : P.blade);
        else c = i === 0 ? P.bladeLo : P.bladeBack;
        rect(g, x0 + i, y, c);
      }
    }
    if (len >= 9 && face !== "edge") rect(g, 50, cy + dir * (len + 3), P.tip); // tip light
  };
  blade(-1); blade(1);

  bladeL.appendChild(g);
  return g;
});

// ---- bubble wake ----
const rnd = a => a[Math.floor(rand() * a.length)];
const startX = 56;
const spawn = (x = startX) => ({
  x, y: cy + Math.round((rand() - 0.5) * 16), c: rnd(P.wake), far: rnd(P.wakeFar),
  rise: rand() < 0.5 ? -1 : 0,
});
let parts = Array.from({ length: 16 }, () => spawn(startX + Math.floor(rand() * 14)));
function drawWake() {
  wakeL.replaceChildren();
  parts.forEach(p => {
    const big = p.x < 62;
    rect(wakeL, p.x, p.y, p.x > 64 ? p.far : p.c, big ? 2 : 1, big ? 2 : 1);
  });
}
drawWake();


// animate propeller frames with SMIL (discrete visibility), 70ms per frame
const n=frames.length, dur=(n*0.07).toFixed(2);
frames.forEach((g,i)=>{ delete g.style.display; g.attrs.visibility=i===0?"visible":"hidden";
  const vals=Array.from({length:n},(_,j)=>j===i?"visible":"hidden").join(";");
  const a=new El("animate"); Object.assign(a.attrs,{attributeName:"visibility",values:vals,dur:dur+"s",repeatCount:"indefinite",calcMode:"discrete"}); g.appendChild(a);});
// wake: drift bubbles to the right
wakeL.children.forEach((r,i)=>{const a=new El("animateTransform");
  Object.assign(a.attrs,{attributeName:"transform",type:"translate",from:"0 0",to:"80 0",dur:"2.2s",begin:`-${(i*0.37%2.2).toFixed(2)}s`,repeatCount:"indefinite"}); r.appendChild(a);
  const o=new El("animate"); Object.assign(o.attrs,{attributeName:"opacity",values:"1;1;0",dur:"2.2s",begin:`-${(i*0.37%2.2).toFixed(2)}s`,repeatCount:"indefinite"}); r.appendChild(o);});
Object.assign(root.attrs,{xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 720 480","shape-rendering":"crispEdges",role:"img","aria-label":"Yellow ki-bekko pixel koi whose tail is a spinning two-blade propeller"});
require("fs").writeFileSync(process.argv[2], root.ser());
