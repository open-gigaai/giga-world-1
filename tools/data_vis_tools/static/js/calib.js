const $ = id => document.getElementById(id);

let defaults = null;
const controls = {
    cam_pos: { labels: ['x', 'y', 'z'], min: -1.5, max: 1.5, step: 0.005 },
    cam_forward: { labels: ['x', 'y', 'z'], min: -1.5, max: 1.5, step: 0.005 },
    left_arm_base_pos: { labels: ['x', 'y', 'z'], min: -1.0, max: 1.0, step: 0.002 },
    right_arm_base_pos: { labels: ['x', 'y', 'z'], min: -1.0, max: 1.0, step: 0.002 },
    camera_offset_local: { labels: ['x', 'y', 'z'], min: -0.5, max: 0.5, step: 0.002 },
};

function setStatus(text) { $('render_status').value = text; }
function setInfo(obj) { $('info').textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2); }

async function post(path, body) {
    const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    return j;
}

async function getJson(path) {
    const r = await fetch(path);
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    return j;
}

function syncNum(id) { $(`${id}_num`).value = $(id).value; }
function syncRange(id) { $(id).value = $(`${id}_num`).value; }

function makeVec(name) {
    const cfg = controls[name];
    const root = $(name);
    root.innerHTML = '';
    defaults[name].forEach((v, i) => {
        const id = `${name}_${i}`;
        root.insertAdjacentHTML('beforeend', `
      <div class="calib_row">
        <label>${name}.${cfg.labels[i]}</label>
        <input id="${id}" type="range" min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" value="${v}">
        <input id="${id}_num" type="number" step="${cfg.step}" value="${v}">
      </div>`);
        $(id).addEventListener('input', () => { syncNum(id); renderAllDebounced(); });
        $(`${id}_num`).addEventListener('change', () => { syncRange(id); renderAllDebounced(); });
    });
}

function makeK() {
    const root = $('K');
    root.innerHTML = '';
    [['fx', 0, 0, 100, 1500, 1], ['fy', 1, 1, 100, 1500, 1], ['cx', 0, 2, -1000, 2000, 1], ['cy', 1, 2, -1000, 2000, 1]].forEach(([label, r, c, min, max, step]) => {
        const id = `K_${r}_${c}`;
        const v = defaults.K[r][c];
        root.insertAdjacentHTML('beforeend', `
      <div class="calib_row">
        <label>K.${label}</label>
        <input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${v}">
        <input id="${id}_num" type="number" step="${step}" value="${v}">
      </div>`);
        $(id).addEventListener('input', () => { syncNum(id); renderAllDebounced(); });
        $(`${id}_num`).addEventListener('change', () => { syncRange(id); renderAllDebounced(); });
    });
}

function vec(name) { return defaults[name].map((_, i) => parseFloat($(`${name}_${i}_num`).value)); }

function params() {
    const K = [
        [parseFloat($('K_0_0_num').value), 0, parseFloat($('K_0_2_num').value)],
        [0, parseFloat($('K_1_1_num').value), parseFloat($('K_1_2_num').value)],
        [0, 0, 1],
    ];
    return {
        K,
        cam_pos: vec('cam_pos'),
        cam_forward: vec('cam_forward'),
        left_arm_base_pos: vec('left_arm_base_pos'),
        right_arm_base_pos: vec('right_arm_base_pos'),
        camera_offset_local: vec('camera_offset_local'),
        ee_link: $('ee_link').value,
        urdf_path: $('urdf_path').value,
    };
}

function payload() {
    return {
        data_pkl: $('data_pkl').value,
        image_key: $('image_key').value,
        save_path: $('save_path').value,
        episode_index: parseInt($('episode_index').value || '0'),
        frame_idx: parseInt($('frame_idx_num').value || '0'),
        params: params(),
    };
}

let renderTimer = null;
let multiCamTimer = null;
function renderAllDebounced() {
    clearTimeout(renderTimer);
    clearTimeout(multiCamTimer);
    renderTimer = setTimeout(renderAll, 160);
    multiCamTimer = setTimeout(renderMultiOverlays, 80);
}

async function loadDefaults() {
    const j = await getJson('/api/calib/defaults');
    defaults = j.params;
    $('save_path').value = j.save_path;
    $('urdf_path').value = defaults.urdf_path;
    for (const key of Object.keys(controls)) makeVec(key);
    makeK();
}

async function loadEpisodes() {
    try {
        setStatus('loading episodes');
        const j = await post('/api/calib/list_episodes', payload());
        $('episode_index').innerHTML = '';
        j.episodes.forEach(ep => $('episode_index').insertAdjacentHTML('beforeend', `<option value="${ep.index}">${ep.index}: ${ep.name} (${ep.qpos_frames})</option>`));
        setInfo(j.episodes.slice(0, 8));
        await renderAll();
    } catch (e) { setStatus('error'); setInfo(e.stack || e.message); }
}

async function renderAll() {
    if (!$('data_pkl').value || !defaults) return;
    try {
        setStatus('rendering');
        const j = await post('/api/calib/render', payload());
        $('frame_idx').max = Math.max(0, Math.min(j.info.total_video_frames, j.info.total_qpos_frames) - 1);
        setInfo(j.info);
        await renderMultiOverlays();
        const p = await post('/api/calib/render3d', payload());
        $('plot3d').src = p.image;
        setStatus('ready');
    } catch (e) { setStatus('error'); setInfo(e.stack || e.message); }
}

async function renderMultiOverlays() {
    if (!$('data_pkl').value) return;
    const count = Math.max(1, Math.min(12, parseInt($('multi_cam_count').value || '4')));
    const stride = Math.max(1, parseInt($('multi_cam_stride').value || '10'));
    const baseFrame = parseInt($('frame_idx_num').value || '0');
    const imageKey = $('image_key').value || 'cam_high_video_path';
    const frameIds = Array.from({ length: count }, (_, i) => Math.max(0, baseFrame + (i - Math.floor(count / 2)) * stride));
    const grid = $('multi_cam_grid');
    grid.innerHTML = frameIds.map((frameId, idx) => `
        <div class="multi_cam_card" id="cam_card_frame_${idx}">
          <div class="multi_cam_title">Overlay | ${imageKey} | frame ${frameId}</div>
          <div class="multi_cam_placeholder">rendering overlay...</div>
        </div>`).join('');

    await Promise.all(frameIds.map(async (frameId, idx) => {
        const card = $(`cam_card_frame_${idx}`);
        try {
            const j = await post('/api/calib/render', { ...payload(), image_key: imageKey, frame_idx: frameId });
            if (j.image) {
                const info = j.info || {};
                card.innerHTML = `<div class="multi_cam_title">Overlay | frame ${info.frame_idx + 1 || frameId}/${info.total_video_frames || '?'}</div><img class="multi_cam_image" src="${j.image}">`;
            } else {
                card.innerHTML = `<div class="multi_cam_title">Overlay | frame ${frameId}</div><div class="multi_cam_placeholder">无 overlay</div>`;
            }
        } catch (e) {
            card.innerHTML = `<div class="multi_cam_title">Overlay | frame ${frameId}</div><div class="multi_cam_placeholder">${e.message}</div>`;
        }
    }));
}

async function saveParams() {
    try {
        const j = await post('/api/calib/save', payload());
        setStatus('saved');
        setInfo(`saved: ${j.save_path}\n${JSON.stringify(j.payload.params, null, 2)}`);
    } catch (e) { setStatus('error'); setInfo(e.stack || e.message); }
}

function resetParams() {
    for (const key of Object.keys(controls)) {
        defaults[key].forEach((v, i) => {
            const id = `${key}_${i}`;
            $(id).value = v;
            $(`${id}_num`).value = v;
        });
    }
    [['K_0_0', defaults.K[0][0]], ['K_1_1', defaults.K[1][1]], ['K_0_2', defaults.K[0][2]], ['K_1_2', defaults.K[1][2]]].forEach(([id, v]) => {
        $(id).value = v;
        $(`${id}_num`).value = v;
    });
    $('ee_link').value = defaults.ee_link;
    $('urdf_path').value = defaults.urdf_path;
    renderAllDebounced();
}

$('back_vis_btn').onclick = () => { window.location.href = '/'; };
$('calib_page_btn').onclick = () => { window.location.href = '/calib'; };
$('load_episodes_btn').onclick = loadEpisodes;
$('render_btn').onclick = renderAll;
$('save_btn').onclick = saveParams;
$('reset_btn').onclick = resetParams;
$('episode_index').addEventListener('change', renderAll);
$('multi_cam_count').addEventListener('change', renderMultiOverlays);
$('multi_cam_stride').addEventListener('change', renderMultiOverlays);
$('image_key').addEventListener('change', renderAllDebounced);
$('frame_idx').addEventListener('input', () => { $('frame_idx_num').value = $('frame_idx').value; renderAllDebounced(); });
$('frame_idx_num').addEventListener('change', () => { $('frame_idx').value = $('frame_idx_num').value; renderAllDebounced(); });

loadDefaults().catch(e => setInfo(e.stack || e.message));
