"""RGB point clouds + colored material-motion segments, inspired by MotionCrafter.

Reference: TencentARC/MotionCrafter/visualize/visualize.py. Our input is already
in a shared fixed frame, so camera extrinsics must NOT be applied a second time.
Produces a standalone interactive HTML (no server/CDN) and a PNG inspection grid.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np


def make_preview(path, output, stride=4, threshold=0.001):
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        n = len(f["frame_index"])
        score = np.zeros(n)
        for key in ("image", "wrist_image"):
            g = f[key]
            for i in range(n - stride):
                flow = g[f"flow_dt{stride}/scene_flow"][i]
                score[i] += np.linalg.norm(flow, axis=-1).sum()
        selected = int(score.argmax())
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        vmax = max(float(np.percentile(np.linalg.norm(f["image"][f"flow_dt{stride}/scene_flow"][selected], axis=-1), 99)), 0.005)
        for row, key in enumerate(("image", "wrist_image")):
            g = f[key]
            rgb = g["rgb"][selected]
            p = g["point_map"][selected]
            v = g[f"flow_dt{stride}/scene_flow"][selected]
            valid = g["valid_mask"][selected]
            flow_valid = g[f"flow_dt{stride}/valid_mask"][selected]
            mag = np.linalg.norm(v, axis=-1)
            ax = fig.add_subplot(2, 4, row * 4 + 1)
            ax.imshow(rgb); ax.set_title(f"{key}: RGB at frame {selected}"); ax.axis("off")
            ax = fig.add_subplot(2, 4, row * 4 + 2)
            im = ax.imshow(np.ma.masked_where(~flow_valid, mag * 1000), cmap="magma", vmin=0, vmax=vmax * 1000)
            ax.set_title(f"3D displacement over {stride} actions (mm)"); ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.7)
            ax = fig.add_subplot(2, 4, row * 4 + 3)
            ax.imshow(g[f"flow_dt{stride}/visible_next"][selected], cmap="gray", vmin=0, vmax=1)
            ax.set_title("Endpoint visible in target view"); ax.axis("off")
            ax = fig.add_subplot(2, 4, row * 4 + 4, projection="3d")
            idx = np.flatnonzero(valid)
            if len(idx) > 12000: idx = idx[np.linspace(0, len(idx) - 1, 12000).astype(int)]
            xyz, colors = p.reshape(-1, 3)[idx], rgb.reshape(-1, 3)[idx] / 255
            ax.set_facecolor("#101827")
            ax.scatter(*xyz.T, c=colors, s=1.1, alpha=0.9)
            dynamic = np.flatnonzero(flow_valid & (mag > threshold))
            if len(dynamic) > 300: dynamic = dynamic[np.linspace(0, len(dynamic) - 1, 300).astype(int)]
            starts, vectors = p.reshape(-1, 3)[dynamic], v.reshape(-1, 3)[dynamic]
            if len(dynamic):
                ax.quiver(*starts.T, *vectors.T, color=plt.cm.hsv(np.linspace(0, 1, len(dynamic))),
                          arrow_length_ratio=0.25, linewidth=0.7)
            ax.set_title("RGB point cloud + 3D motion (true scale)")
            ax.view_init(elev=-70, azim=-90)
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
        fig.suptitle("LIBERO replay: same surface point displacement in fixed initial-camera coordinates", fontsize=14)
        png = output / "motion_preview.png"
        fig.savefig(png, dpi=160)
        plt.close(fig)
    return selected, png


def make_html(path, output, stride=4, threshold=0.001, max_points=3500, max_lines=500):
    import h5py
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    with h5py.File(path, "r") as f:
        center, scale = f["normalization/center"][:], float(f["normalization/scale"][()])
        for i in range(0, len(f["frame_index"]), stride):
            views = []
            for key in ("image", "wrist_image"):
                g = f[key]
                rgb = g["rgb"][i]
                p = (g["point_map"][i] - center) / scale
                v = g[f"flow_dt{stride}/scene_flow"][i] / scale
                valid = g["valid_mask"][i]
                idx = np.flatnonzero(valid)
                if len(idx) > max_points: idx = idx[np.linspace(0, len(idx) - 1, max_points).astype(int)]
                dynamic = np.flatnonzero(g[f"flow_dt{stride}/valid_mask"][i] & (np.linalg.norm(v, axis=-1) * scale > threshold))
                if len(dynamic) > max_lines: dynamic = dynamic[np.linspace(0, len(dynamic) - 1, max_lines).astype(int)]
                starts = p.reshape(-1, 3)[dynamic]
                ends = starts + v.reshape(-1, 3)[dynamic]
                colors = (plt.cm.hsv(np.linspace(0, 1, len(dynamic)))[:, :3] * 255).astype(np.uint8)
                image = Image.fromarray(rgb).resize((192, 192))
                buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=80)
                views.append(dict(name=key, p=np.round(p.reshape(-1, 3)[idx], 4).tolist(),
                    c=rgb.reshape(-1, 3)[idx].tolist(), a=np.round(starts, 4).tolist(),
                    b=np.round(ends, 4).tolist(), lc=colors.tolist(),
                    rgb=base64.b64encode(buffer.getvalue()).decode()))
            frames.append(dict(index=i, views=views))
        meta = json.loads(f.attrs["report_json"])
    payload = json.dumps(dict(frames=frames, meta=meta), separators=(",", ":"))
    html = HTML.replace("__PAYLOAD__", payload).replace("__STRIDE__", str(stride))
    result = Path(output) / "motion_viewer.html"
    result.write_text(html)
    return result


HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>LIBERO · 3D Motion</title><style>
*{box-sizing:border-box}body{margin:0;background:#101827;color:#edf4ff;font:15px system-ui}
main{max-width:1280px;margin:auto;padding:24px}h1{font-size:24px;margin:0 0 8px}
p{color:#b6c6db;margin:8px 0 20px}canvas{width:100%;height:620px;background:#080f1b;border-radius:12px;touch-action:none}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:18px 0}input[type=range]{width:260px}
button,select{border:1px solid #49627d;border-radius:6px;background:#203248;color:white;padding:8px}
.thumbs{display:flex;gap:18px}.thumbs img{width:170px;border-radius:8px}.caption{color:#b6c6db;font-size:13px}
</style><main><h1>LIBERO · RGB 点云与三维运动</h1>
<p>与 MotionCrafter 相同的表示：固定坐标系内的点云，加上同一表面点的位移线段。拖动旋转，滚轮缩放。此文件可离线打开。</p>
<div class="controls"><button id="play">播放</button><input id="time" type="range" min="0" value="0"><span id="label"></span>
<select id="view"><option value="both">两路相机</option><option value="0">外部相机</option><option value="1">手腕相机</option></select>
<label><input id="flow" type="checkbox" checked>显示 motion</label><label>线段倍率 <input id="gain" type="number" min="1" max="20" value="1" style="width:50px"></label>
<button id="reset">重置视角</button></div><canvas id="scene"></canvas>
<div class="thumbs"><div><img id="rgb0"><div class="caption">外部相机 RGB</div></div><div><img id="rgb1"><div class="caption">手腕相机 RGB</div></div></div>
<p id="meta" class="caption"></p></main><script>
const D=__PAYLOAD__,F=D.frames,C=document.querySelector('#scene'),ctx=C.getContext('2d');
const slider=document.querySelector('#time');slider.max=F.length-1;
let yaw=0,pitch=Math.PI,zoom=230,playing=false,drag=null;
function project(p){let x=p[0],y=p[1],z=p[2];let a=Math.cos(yaw),b=Math.sin(yaw),c=Math.cos(pitch),d=Math.sin(pitch);let xx=a*x+b*z,zz=-b*x+a*z;let yy=c*y-d*zz,depth=d*y+c*zz;let s=zoom*(window.devicePixelRatio||1)*5/Math.max(1.2,5+depth);return [C.width/2+xx*s,C.height/2+yy*s,depth]}
function draw(){let rect=C.getBoundingClientRect(),ratio=window.devicePixelRatio||1;C.width=rect.width*ratio;C.height=rect.height*ratio;ctx.clearRect(0,0,C.width,C.height);
 let f=F[+slider.value],view=document.querySelector('#view').value,show=document.querySelector('#flow').checked,gain=+document.querySelector('#gain').value;
 document.querySelector('#label').textContent=`帧 ${f.index} · dt=__STRIDE__ 数据步`;
 document.querySelector('#rgb0').src='data:image/jpeg;base64,'+f.views[0].rgb;document.querySelector('#rgb1').src='data:image/jpeg;base64,'+f.views[1].rgb;
 let dots=[];f.views.forEach((v,k)=>{if(view!=='both'&&view!==''+k)return;v.p.forEach((p,j)=>dots.push([project(p),v.c[j]]))});dots.sort((a,b)=>b[0][2]-a[0][2]);
 dots.forEach(([p,c])=>{ctx.fillStyle=`rgb(${c})`;ctx.globalAlpha=.8;ctx.fillRect(p[0],p[1],2*ratio,2*ratio)});ctx.globalAlpha=1;
 if(show)f.views.forEach((v,k)=>{if(view!=='both'&&view!==''+k)return;v.a.forEach((a,j)=>{let b=v.b[j].map((x,i)=>a[i]+gain*(x-a[i])),p=project(a),q=project(b);ctx.strokeStyle=`rgb(${v.lc[j]})`;ctx.lineWidth=1.4*ratio;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()})});}
slider.oninput=draw;document.querySelector('#view').onchange=draw;document.querySelector('#flow').onchange=draw;document.querySelector('#gain').onchange=draw;
document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'暂停':'播放'};
document.querySelector('#reset').onclick=()=>{yaw=0;pitch=Math.PI;zoom=230;draw()};
C.onpointerdown=e=>{drag=[e.clientX,e.clientY];C.setPointerCapture(e.pointerId)};C.onpointerup=()=>drag=null;
C.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw()};
C.onwheel=e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*.001);draw()};window.onresize=draw;
setInterval(()=>{if(playing){slider.value=(+slider.value+1)%F.length;draw()}},200);
document.querySelector('#meta').textContent=`Episode ${D.meta.episode} · ${D.meta.frames} 帧 · 两路共享第一帧外部相机坐标 · 单一中心和尺度归一化 · flow 不减均值、不截断。线段倍率 1 表示真实位移。`;
draw();</script></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--threshold-mm", type=float, default=1)
    a = p.parse_args()
    selected, png = make_preview(a.motion, a.output_dir, a.stride, a.threshold_mm / 1000)
    html = make_html(a.motion, a.output_dir, a.stride, a.threshold_mm / 1000)
    print(json.dumps(dict(selected_frame=selected, png=str(png), html=str(html)), indent=2))


if __name__ == "__main__":
    main()
