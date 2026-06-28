import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

const $ = id => document.getElementById(id);
const JOINT_COLORS = ['#ff6666', '#66d9ef', '#a6e22e', '#fd971f', '#ae81ff', '#ffd866', '#f92672', '#66ffcc'];

async function loadDefaults() {
  try {
    const res = await fetch('/api/defaults');
    const body = await res.json();
    if (body.urdf_path) $('urdf_path').value = body.urdf_path;
  } catch (_) {
    if (window.DEFAULT_URDF_PATH) $('urdf_path').value = window.DEFAULT_URDF_PATH;
  }
}
loadDefaults();

let scene, camera, renderer, controls;
let leftRoot, rightRoot, robotModel, dataPayload;
let leftLinkGroups = {}, rightLinkGroups = {};
let modelMaterials = [];
let leftSkeleton, rightSkeleton, leftJointAxes, rightJointAxes;
let playing = false, timer = null, currentFrame = 0;
let manualLeft = null, manualRight = null;
let mediaRecorder = null, recordedChunks = [];
let cameraPreviewBusy = false, cameraPreviewPending = false, lastCameraPreviewTime = 0;
let activeJointSlider = null;
let stlLoader = new STLLoader();

initThree();
animate();

function initThree() {
  const viewer = $('viewer');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);
  scene.up.set(0, 0, 1);
  camera = new THREE.PerspectiveCamera(55, viewer.clientWidth / viewer.clientHeight, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(1.05, -1.15, 0.58);
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
  viewer.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.25, 0.0, 0.22);
  controls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 1.25));
  const light = new THREE.DirectionalLight(0xffffff, 1.05);
  light.position.set(1.5, -2.0, 3.0);
  scene.add(light);
  const grid = new THREE.GridHelper(2.0, 20, 0x64748b, 0x263044);
  grid.name = 'ground_grid';
  grid.rotation.x = Math.PI / 2;
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  scene.add(grid);
  scene.add(new THREE.AxesHelper(0.35));

  window.addEventListener('resize', () => {
    camera.aspect = viewer.clientWidth / viewer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewer.clientWidth, viewer.clientHeight);
    drawJointCurves();
  });
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function setStatus(text) { $('status').textContent = text; }
function setInfo(obj) { $('info').textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2); }
function num(id) { return parseFloat($(id).value); }

function params() {
  return {
    data_pkl: $('data_pkl').value,
    urdf_path: $('urdf_path').value,
    source_key: $('source_key').value,
    episode_index: parseInt($('episode_index').value || '0'),
    ee_link: $('ee_link').value || 'link6',
    left_base: [num('lbx'), num('lby'), num('lbz')],
    right_base: [num('rbx'), num('rby'), num('rbz')],
  };
}

async function loadScene() {
  pause(false);
  setStatus('加载中...');
  const res = await fetch('/api/scene', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params()) });
  const body = await res.json();
  if (!res.ok || body.error) throw new Error(body.error || res.statusText);
  dataPayload = body;
  robotModel = body.robot;
  await buildRobotMeshes();
  buildJointControls();
  buildCurveLegends();
  $('frame_slider').max = Math.max(0, body.frame_count - 1);
  $('frame_num').max = Math.max(0, body.frame_count - 1);
  setFrame(0);
  updateCameraPreview(true);
  setDefaultView();
  setStatus(`已加载 ${body.frame_count} 帧`);
  setInfo({ episode: body.episode, source_key: body.source_key, frame_count: body.frame_count, total_episodes: body.total_episodes });
}

function clearRobot() {
  if (leftRoot) scene.remove(leftRoot);
  if (rightRoot) scene.remove(rightRoot);
  if (leftSkeleton) scene.remove(leftSkeleton);
  if (rightSkeleton) scene.remove(rightSkeleton);
  if (leftJointAxes) scene.remove(leftJointAxes);
  if (rightJointAxes) scene.remove(rightJointAxes);
  leftRoot = new THREE.Group();
  rightRoot = new THREE.Group();
  leftSkeleton = new THREE.Group();
  rightSkeleton = new THREE.Group();
  leftJointAxes = new THREE.Group();
  rightJointAxes = new THREE.Group();
  scene.add(leftRoot, rightRoot, leftSkeleton, rightSkeleton, leftJointAxes, rightJointAxes);
  leftLinkGroups = {};
  rightLinkGroups = {};
  modelMaterials = [];
  manualLeft = null;
  manualRight = null;
}

async function buildRobotMeshes() {
  clearRobot();
  leftRoot.applyMatrix4(matFromList(dataPayload.left_base_T));
  rightRoot.applyMatrix4(matFromList(dataPayload.right_base_T));
  for (const side of ['left', 'right']) {
    const root = side === 'left' ? leftRoot : rightRoot;
    const map = side === 'left' ? leftLinkGroups : rightLinkGroups;
    for (const link of robotModel.links) {
      const g = new THREE.Group();
      g.name = `${side}:${link.name}`;
      root.add(g);
      map[link.name] = g;
      for (const vis of link.visuals) await addVisual(g, vis, side);
    }
  }
}

function addVisual(parent, vis, side) {
  return new Promise(resolve => {
    stlLoader.load(vis.url, geom => {
      geom.computeVertexNormals();
      const rgba = vis.color || [0.78, 0.82, 0.93, 1.0];
      const color = side === 'left'
        ? new THREE.Color(rgba[0] * 0.75, Math.min(1, rgba[1] * 1.08), rgba[2] * 0.75)
        : new THREE.Color(Math.min(1, rgba[0] * 1.08), rgba[1] * 0.72, rgba[2] * 0.45);
      const transparent = $('transparent_model').checked;
      const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.65, metalness: 0.05, transparent: true, opacity: transparent ? 0.46 : 1.0, depthWrite: !transparent });
      modelMaterials.push(mat);
      const mesh = new THREE.Mesh(geom, mat);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(matFromList(vis.origin_T));
      mesh.scale.set(vis.scale[0], vis.scale[1], vis.scale[2]);
      parent.add(mesh);
      resolve();
    }, undefined, () => resolve());
  });
}

function matFromList(list) {
  const m = new THREE.Matrix4();
  const e = list.flat();
  m.set(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], e[12], e[13], e[14], e[15]);
  return m;
}
function identity() { return new THREE.Matrix4(); }
function axisAngle(axis, q) { return new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(axis[0], axis[1], axis[2]).normalize(), q); }
function prismatic(axis, q) { return new THREE.Matrix4().makeTranslation(axis[0] * q, axis[1] * q, axis[2] * q); }
function jointMotion(joint, q) {
  if (joint.type === 'revolute' || joint.type === 'continuous') return axisAngle(joint.axis, q || 0);
  if (joint.type === 'prismatic') return prismatic(joint.axis, q || 0);
  return identity();
}

function applyFK(side, qpos) {
  const map = side === 'left' ? leftLinkGroups : rightLinkGroups;
  if (!robotModel || !map[robotModel.base_link]) return [];
  let T = identity();
  map[robotModel.base_link].matrix.copy(T);
  map[robotModel.base_link].matrixAutoUpdate = false;
  const jointWorld = [{ name: robotModel.base_link, T: T.clone(), movable: false }];
  let qi = 0;
  for (const joint of robotModel.joints) {
    T = T.clone().multiply(matFromList(joint.origin_T));
    const movable = joint.type === 'revolute' || joint.type === 'continuous' || joint.type === 'prismatic';
    if (movable) {
      T = T.clone().multiply(jointMotion(joint, qpos[qi] || 0));
      qi += 1;
    }
    jointWorld.push({ name: joint.child, T: T.clone(), movable });
    if (map[joint.child]) {
      map[joint.child].matrixAutoUpdate = false;
      map[joint.child].matrix.copy(T);
    }
  }
  return jointWorld;
}

function transformJointWorld(side, jointWorld) {
  const base = side === 'left' ? matFromList(dataPayload.left_base_T) : matFromList(dataPayload.right_base_T);
  return jointWorld.map(item => ({ ...item, T: base.clone().multiply(item.T) }));
}
function positionFromMatrix(T) { const p = new THREE.Vector3(); p.setFromMatrixPosition(T); return p; }

function drawSkeleton(group, jointWorld, color) {
  group.clear();
  group.visible = $('show_skeleton').checked;
  const pts = jointWorld.map(item => positionFromMatrix(item.T));
  if (pts.length >= 2) group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color })));
  for (const p of pts) {
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), new THREE.MeshBasicMaterial({ color }));
    marker.position.copy(p);
    group.add(marker);
  }
}

function makeAxisLabel(text, color, pos) {
  const canvas = document.createElement('canvas');
  canvas.width = 64; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 42px Arial'; ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, 32, 34);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthTest: false }));
  sprite.position.copy(pos); sprite.scale.set(0.06, 0.06, 0.06);
  return sprite;
}
function makeThickAxis(T) {
  const g = new THREE.Group();
  const axes = [
    { name: 'X', color: 0xff3333, css: '#ff5555', dir: new THREE.Vector3(1, 0, 0) },
    { name: 'Y', color: 0x33ff33, css: '#55ff55', dir: new THREE.Vector3(0, 1, 0) },
    { name: 'Z', color: 0x3377ff, css: '#5590ff', dir: new THREE.Vector3(0, 0, 1) },
  ];
  const len = 0.09, radius = 0.0045;
  for (const a of axes) {
    const end = a.dir.clone().multiplyScalar(len);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), end]), new THREE.LineBasicMaterial({ color: a.color, linewidth: 4, depthTest: false }));
    line.renderOrder = 999; g.add(line);
    const cyl = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, len, 10), new THREE.MeshBasicMaterial({ color: a.color, depthTest: false }));
    cyl.position.copy(end.clone().multiplyScalar(0.5));
    cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), a.dir.clone().normalize());
    g.add(cyl);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(radius * 2.2, 0.018, 12), new THREE.MeshBasicMaterial({ color: a.color, depthTest: false }));
    cone.position.copy(end);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), a.dir.clone().normalize());
    g.add(cone);
    g.add(makeAxisLabel(a.name, a.css, a.dir.clone().multiplyScalar(len + 0.035)));
  }
  g.matrixAutoUpdate = false; g.matrix.copy(T);
  return g;
}
function drawJointAxes(group, jointWorld) {
  group.clear();
  const visible = $('show_joint_axes').checked;
  group.visible = visible;
  $('axis_legend').classList.toggle('hidden', !visible);
  for (const item of jointWorld) if (item.movable) group.add(makeThickAxis(item.T));
}

function updateJointControls(leftQ, rightQ) {
  for (const [side, q] of [['left', leftQ], ['right', rightQ]]) {
    for (let i = 0; i < q.length; i++) {
      const slider = $(`${side}_joint_${i}`);
      const val = $(`${side}_joint_${i}_val`);
      const id = `${side}_${i}`;
      if (slider && activeJointSlider !== id) slider.value = q[i];
      if (val && activeJointSlider !== id) val.value = Number(q[i]).toFixed(3);
    }
  }
}
function currentQ() {
  return {
    leftQ: manualLeft || (dataPayload ? dataPayload.left[currentFrame].slice() : []),
    rightQ: manualRight || (dataPayload ? dataPayload.right[currentFrame].slice() : []),
  };
}
function refreshPose() {
  if (!dataPayload) return;
  const { leftQ, rightQ } = currentQ();
  const leftWorld = transformJointWorld('left', applyFK('left', leftQ));
  const rightWorld = transformJointWorld('right', applyFK('right', rightQ));
  drawSkeleton(leftSkeleton, leftWorld, 0x62d08f);
  drawSkeleton(rightSkeleton, rightWorld, 0xffa15a);
  drawJointAxes(leftJointAxes, leftWorld);
  drawJointAxes(rightJointAxes, rightWorld);
  updateJointControls(leftQ, rightQ);
}
function updateModelTransparency() {
  const transparent = $('transparent_model').checked;
  for (const mat of modelMaterials) { mat.transparent = true; mat.opacity = transparent ? 0.46 : 1.0; mat.depthWrite = !transparent; mat.needsUpdate = true; }
}

async function updateCameraPreview(force = false) {
  if (!$('show_camera_preview').checked) { $('camera_preview').classList.add('hidden'); return; }
  const now = performance.now();
  if (!force && playing && now - lastCameraPreviewTime < 250) return;
  lastCameraPreviewTime = now;
  if (!dataPayload || cameraPreviewBusy) { cameraPreviewPending = true; return; }
  cameraPreviewBusy = true;
  cameraPreviewPending = false;
  try {
    const res = await fetch('/api/camera_frame', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...params(), frame_idx: currentFrame }) });
    const body = await res.json();
    if (body.image) {
      $('camera_preview_img').src = body.image;
      $('camera_preview_title').textContent = `${body.video_key} | ${body.frame_idx + 1}/${body.total_frames}`;
      $('camera_preview').classList.remove('hidden');
    } else $('camera_preview').classList.add('hidden');
  } catch { $('camera_preview').classList.add('hidden'); }
  finally { cameraPreviewBusy = false; if (cameraPreviewPending) updateCameraPreview(); }
}

function setFrame(i) {
  if (!dataPayload) return;
  manualLeft = null; manualRight = null;
  currentFrame = Math.max(0, Math.min(dataPayload.frame_count - 1, parseInt(i || 0)));
  $('frame_slider').value = currentFrame;
  $('frame_num').value = currentFrame;
  refreshPose(); updateCameraPreview(); drawJointCurves();
  setStatus(`${playing ? '播放' : '暂停'} | frame ${currentFrame + 1}/${dataPayload.frame_count}`);
}

function resizeCanvasToDisplaySize(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  return { width, height, dpr };
}
function drawJointCurve(canvasId, frames, title) {
  const canvas = $(canvasId); if (!canvas) return;
  const { width, height, dpr } = resizeCanvasToDisplaySize(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height); ctx.fillStyle = '#050816'; ctx.fillRect(0, 0, width, height);
  ctx.save(); ctx.scale(dpr, dpr);
  const w = width / dpr, h = height / dpr;
  const padL = 28, padR = 8, padT = 18, padB = 18;
  const plotW = Math.max(1, w - padL - padR), plotH = Math.max(1, h - padT - padB);
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 1; ctx.strokeRect(padL, padT, plotW, plotH);
  for (let k = 1; k < 4; k++) { const y = padT + plotH * k / 4; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke(); }
  ctx.fillStyle = '#cbd5e1'; ctx.font = '12px Arial'; ctx.fillText(title, padL, 12);
  if (!frames || frames.length === 0) { ctx.fillText('无数据', padL + 8, padT + 24); ctx.restore(); return; }
  const jointCount = Math.max(...frames.map(f => (f || []).length));
  const n = frames.length;
  for (let j = 0; j < jointCount; j++) {
    const vals = frames.map(f => Number((f || [])[j] || 0));
    const mn = Math.min(...vals), mx = Math.max(...vals);
    if (!Number.isFinite(mn) || !Number.isFinite(mx)) continue;
    const range = Math.max(1e-9, mx - mn);
    ctx.strokeStyle = JOINT_COLORS[j % JOINT_COLORS.length]; ctx.lineWidth = 1.45; ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padL + (n <= 1 ? 0 : i / (n - 1)) * plotW;
      const y = padT + (1 - (vals[i] - mn) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  const cx = padL + (n <= 1 ? 0 : currentFrame / (n - 1)) * plotW;
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(cx, padT); ctx.lineTo(cx, padT + plotH); ctx.stroke();
  ctx.fillStyle = '#94a3b8'; ctx.font = '11px Arial'; ctx.fillText('0', 4, padT + plotH + 3); ctx.fillText('1', 4, padT + 4); ctx.fillText(`${currentFrame + 1}/${n}`, padL + plotW - 48, h - 4);
  ctx.restore();
}
function drawJointCurves() {
  if (!dataPayload) return;
  const leftFrames = dataPayload.left.map(f => f.slice());
  const rightFrames = dataPayload.right.map(f => f.slice());
  if (manualLeft) leftFrames[currentFrame] = manualLeft.slice();
  if (manualRight) rightFrames[currentFrame] = manualRight.slice();
  drawJointCurve('left_joint_curve', leftFrames, '左臂 left');
  drawJointCurve('right_joint_curve', rightFrames, '右臂 right');
}
function buildCurveLegends() {
  const names = robotModel.joints.filter(j => j.type === 'revolute' || j.type === 'continuous' || j.type === 'prismatic').map(j => j.name);
  for (const side of ['left', 'right']) {
    const box = $(`${side}_curve_legend`);
    box.innerHTML = names.map((name, i) => `<div class="legend_item"><span class="legend_swatch" style="background:${JOINT_COLORS[i % JOINT_COLORS.length]}"></span>${name}</div>`).join('');
  }
}

function setManualJoint(side, i, value) {
  if (!dataPayload) return;
  value = Number(value);
  if (!Number.isFinite(value)) return;
  if (!manualLeft || !manualRight) { manualLeft = dataPayload.left[currentFrame].slice(); manualRight = dataPayload.right[currentFrame].slice(); }
  const q = side === 'left' ? manualLeft : manualRight;
  q[i] = value;
  const slider = $(`${side}_joint_${i}`), val = $(`${side}_joint_${i}_val`);
  if (slider) slider.value = value;
  if (val) val.value = Number(value).toFixed(3);
  refreshPose(); drawJointCurves();
  setStatus(`手动调整 | frame ${currentFrame + 1}/${dataPayload.frame_count}`);
}
function buildJointControls() {
  const movableNames = robotModel.joints.filter(j => j.type === 'revolute' || j.type === 'continuous' || j.type === 'prismatic').map(j => j.name);
  for (const side of ['left', 'right']) {
    const box = $(`${side}_joint_controls`); box.innerHTML = '';
    movableNames.forEach((name, i) => {
      const row = document.createElement('div'); row.className = 'joint_row';
      row.innerHTML = `<span>${name}</span><input id="${side}_joint_${i}" type="range" min="-3.2" max="3.2" step="0.001"><input id="${side}_joint_${i}_val" class="joint_val" type="number" min="-3.2" max="3.2" step="0.001" value="0.000">`;
      box.appendChild(row);
      const slider = row.querySelector(`#${side}_joint_${i}`), val = row.querySelector(`#${side}_joint_${i}_val`);
      slider.addEventListener('pointerdown', () => { activeJointSlider = `${side}_${i}`; pause(false); });
      slider.addEventListener('pointerup', () => { activeJointSlider = null; });
      slider.addEventListener('touchend', () => { activeJointSlider = null; });
      slider.addEventListener('input', e => { activeJointSlider = `${side}_${i}`; pause(false); setManualJoint(side, i, parseFloat(e.target.value)); });
      slider.addEventListener('change', () => { activeJointSlider = null; });
      val.addEventListener('focus', () => { activeJointSlider = `${side}_${i}`; pause(false); });
      val.addEventListener('input', e => { activeJointSlider = `${side}_${i}`; pause(false); setManualJoint(side, i, Math.max(-3.2, Math.min(3.2, parseFloat(e.target.value || '0')))); });
      val.addEventListener('blur', () => { activeJointSlider = null; refreshPose(); });
    });
  }
}

function step(delta) { setFrame(currentFrame + delta); }
function play() {
  if (!dataPayload || playing) return;
  playing = true;
  const tick = () => {
    const fps = Math.max(1, Math.min(60, parseFloat($('fps').value || '15')));
    step(1);
    if (currentFrame >= dataPayload.frame_count - 1) setFrame(0);
    timer = setTimeout(tick, 1000 / fps);
  };
  tick();
}
function pause(resetManual = true) {
  playing = false;
  if (timer) clearTimeout(timer);
  timer = null;
  if (resetManual && dataPayload) setFrame(currentFrame);
}
function setView(pos) { camera.position.set(pos[0], pos[1], pos[2]); controls.target.set(0.25, 0.0, 0.22); controls.update(); }
function setDefaultView() { setView([1.05, -1.15, 0.58]); }
function screenshot() {
  renderer.render(scene, camera);
  const a = document.createElement('a');
  a.href = renderer.domElement.toDataURL('image/png');
  a.download = `urdf_frame_${String(currentFrame).padStart(5, '0')}.png`;
  a.click();
}
function startRecord() {
  if (mediaRecorder && mediaRecorder.state === 'recording') return;
  recordedChunks = [];
  const stream = renderer.domElement.captureStream(Math.max(1, parseFloat($('fps').value || '15')));
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'urdf_animation.webm';
    a.click();
    URL.revokeObjectURL(a.href);
  };
  mediaRecorder.start(); play(); setStatus('录制中...');
}
function stopRecord() { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); pause(); }

$('load_btn').onclick = () => loadScene().catch(err => { setStatus('加载失败'); setInfo(err.stack || err.message); });
$('viewer_page_btn').onclick = () => { window.location.href = '/'; };
$('open_calib_btn').onclick = () => { window.location.href = '/calib'; };
$('prev_btn').onclick = () => { pause(false); step(-1); };
$('next_btn').onclick = () => { pause(false); step(1); };
$('play_btn').onclick = play;
$('pause_btn').onclick = () => pause();
$('frame_slider').oninput = e => { pause(false); setFrame(e.target.value); };
$('frame_num').onchange = e => { pause(false); setFrame(e.target.value); };
$('view_front').onclick = () => setView([1.15, 0.0, 0.35]);
$('view_side').onclick = () => setView([0.25, -1.15, 0.35]);
$('view_top').onclick = () => setView([0.25, 0.0, 1.55]);
$('view_iso').onclick = setDefaultView;
$('reset_view').onclick = setDefaultView;
$('screenshot_btn').onclick = screenshot;
$('record_btn').onclick = startRecord;
$('stop_record_btn').onclick = stopRecord;
$('show_skeleton').onchange = refreshPose;
$('show_joint_axes').onchange = refreshPose;
$('transparent_model').onchange = updateModelTransparency;
$('show_camera_preview').onchange = () => updateCameraPreview(true);
window.addEventListener('keydown', e => {
  if (e.code === 'Space') { playing ? pause() : play(); e.preventDefault(); }
  if (e.code === 'ArrowLeft') { pause(false); step(-1); }
  if (e.code === 'ArrowRight') { pause(false); step(1); }
});
