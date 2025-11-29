// static/js/scene_gallery.js

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { OutlinePass } from "three/addons/postprocessing/OutlinePass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

// ================= 配置区域 =================
const HDR_PATH = 'static/textures/environment.hdr';

const SCENE_IDS = [10,32,24,161,51,141,52,74,72,160,55,70,63,95,36,17,164,60,];

// GitHub Release 基础路径
const BASE_URL = "https://huggingface.co/xinjue1/TabletopGen-GLB-PageDemo/resolve/main/";

// 自动生成场景配置
function generateScenes(ids) {
    return ids.map((id, index) => ({
        id: index+1,
        glb: `static/models/scene_${id}.glb`,
        img: `static/images/models/scene_${id}.png`
    }));
}

const SCENES = generateScenes(SCENE_IDS);

// DOM 元素
const container = document.getElementById("viewer3d");
const loaderText = document.getElementById("loader-text");
const refImage = document.getElementById("ref-image");
const sceneIdOverlay = document.getElementById("scene-id-overlay");
const track = document.getElementById("scene-track");
const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");

// 全局变量
let scene, camera, renderer, controls, composer, outlinePass;
let loadedModel = null;
let nameLabel = null;
let activeIndex = 0; // 当前选中的场景索引 (0-9)
let currentPage = 0; // 当前轮播页码 (0, 1, 2...)
const ITEMS_PER_PAGE = 6; // 每次显示6个

if (container) {
    createNameLabel();
    initViewer();
    initCustomCarousel(); // 初始化我们手写的轮播
}

// --- 1. 初始化 3D 视图 (保持不变,只做微调) ---
function initViewer() {
    let width = container.clientWidth;
    let height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000);
    camera.position.set(0, 5, 8);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    
    container.appendChild(renderer.domElement);

    // 后处理
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    outlinePass = new OutlinePass(new THREE.Vector2(width, height), scene, camera);
    outlinePass.edgeStrength = 3; 
    outlinePass.edgeGlow = 0.5;
    outlinePass.visibleEdgeColor.set('#ffffff');
    outlinePass.hiddenEdgeColor.set('#190a05');
    composer.addPass(outlinePass);
    composer.addPass(new OutputPass());

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };

    // 加载环境和初始场景
    new RGBELoader().load(HDR_PATH, (texture) => {
        texture.mapping = THREE.EquirectangularReflectionMapping;
        scene.environment = texture;
        loadScene(0); // 初始加载第1个
    });

    const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
    dirLight.position.set(5, 10, 8);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.set(2048, 2048);
    scene.add(dirLight);

    // renderer.domElement.addEventListener('click', onCanvasClick);
    // window.addEventListener("resize", onWindowResize);
    // animate();

    renderer.domElement.addEventListener('click', onCanvasClick);

    // 1）窗口变化也更新（可留着）
    window.addEventListener("resize", onWindowResize);

    // 2）容器尺寸变化也更新（关键）
    const resizeObserver = new ResizeObserver(() => {
    onWindowResize();
    });
    resizeObserver.observe(container);

    animate();

}

// --- 2. 加载场景逻辑 (更新) ---
function loadScene(index) {
    if (index < 0 || index >= SCENES.length) return;
    activeIndex = index;
    const data = SCENES[index];

    // 更新 UI
    if (refImage) refImage.src = data.img;
    if (sceneIdOverlay) sceneIdOverlay.innerText = `Scene ${data.id}`; // 只显示 ID
    if (loaderText) {
        loaderText.style.display = 'block';
        loaderText.innerText = `Loading Scene ${data.id}...`;
    }
    
    // 更新缩略图的高亮状态
    updateCarouselHighlight();

    // 清理旧模型
    if (loadedModel) {
        scene.remove(loadedModel);
        loadedModel.traverse(c => {
            if (c.geometry) c.geometry.dispose();
            if (c.material) {
                if(Array.isArray(c.material)) c.material.forEach(m=>m.dispose());
                else c.material.dispose();
            }
        });
        loadedModel = null;
    }
    outlinePass.selectedObjects = [];
    hideLabel();

    // 加载新模型
    const loader = new GLTFLoader();
    // !!! 新增：配置 Draco 解码器 !!!
    const dracoLoader = new DRACOLoader();
    // 使用 CDN 路径，对应 three.js 版本 0.160.0
    dracoLoader.setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/draco/');
    dracoLoader.setDecoderConfig({ type: 'js' }); // 强制使用 JS 版解码器以获得更好兼容性
    loader.setDRACOLoader(dracoLoader);

    loader.load(data.glb, (gltf) => {
        if (loaderText) loaderText.style.display = 'none';
        loadedModel = gltf.scene;

        loadedModel.traverse(c => {
            if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; }
        });
        scene.add(loadedModel);

        // 居中 & 适配
        const box = new THREE.Box3().setFromObject(loadedModel);
        const center = box.getCenter(new THREE.Vector3());
        loadedModel.position.sub(center);
        fitCamera(loadedModel);

        // 加载完成后销毁 dracoLoader 释放内存
        dracoLoader.dispose();

    }, undefined, (err) => {
        console.error(err);
        if(loaderText) loaderText.innerText = "Load Failed";
    });
}

// --- 3. 自定义轮播逻辑 (关键修复) ---
function initCustomCarousel() {
    if (!track) return;

    // A. 动态生成 DOM
    track.innerHTML = '';
    SCENES.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = 'scene-card';
        card.innerHTML = `
            <div class="scene-card-inner">
                <img src="${item.img}" alt="Scene ${item.id}">
            </div>
        `;
        card.onclick = () => loadScene(idx);
        track.appendChild(card);
    });

    // B. 绑定按钮事件
    // 计算总页数
    const totalPages = Math.ceil(SCENES.length / ITEMS_PER_PAGE);

    function updateTrackPosition() {
        // 移动多少百分比? 每一页移动 100%
        const percent = -(currentPage * 100);
        track.style.transform = `translateX(${percent}%)`;
        
        // 更新按钮状态
        btnPrev.classList.toggle('disabled', currentPage === 0);
        btnNext.classList.toggle('disabled', currentPage >= totalPages - 1);
    }

    btnPrev.onclick = () => {
        if (currentPage > 0) {
            currentPage--;
            updateTrackPosition();
        }
    };

    btnNext.onclick = () => {
        if (currentPage < totalPages - 1) {
            currentPage++;
            updateTrackPosition();
        }
    };

    // 初始化状态
    updateTrackPosition();
}

function updateCarouselHighlight() {
    const cards = document.querySelectorAll('.scene-card');
    cards.forEach((card, idx) => {
        if (idx === activeIndex) card.classList.add('is-active');
        else card.classList.remove('is-active');
    });
}

// --- 4. 其他辅助函数 ---
function createNameLabel() {
    nameLabel = document.createElement('div');
    Object.assign(nameLabel.style, {
        position: 'absolute', top: '20px', left: '50%', transform: 'translateX(-50%)',
        backgroundColor: 'rgba(0,0,0,0.8)', color: '#fff', padding: '6px 14px',
        borderRadius: '20px', fontSize: '13px', pointerEvents: 'none', opacity: '0', transition: 'opacity 0.2s', zIndex: '10'
    });
    container.appendChild(nameLabel);
}

function onCanvasClick(event) {
    if (!loadedModel) return;
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(loadedModel, true);

    if (intersects.length > 0) {
        const obj = intersects[0].object;
        outlinePass.selectedObjects = [obj];
        let name = obj.name || "Unknown";
        name = name.replace(/_/g, ' ');
        nameLabel.innerText = name;
        nameLabel.style.opacity = '1';
        nameLabel.style.transform = 'translateX(-50%) translateY(0)';
    } else {
        outlinePass.selectedObjects = [];
        hideLabel();
    }
}

function hideLabel() {
    if (nameLabel) {
        nameLabel.style.opacity = '0';
        nameLabel.style.transform = 'translateX(-50%) translateY(-10px)';
    }
}

function fitCamera(selection) {
    const box = new THREE.Box3().setFromObject(selection);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxSize = Math.max(size.x, size.y, size.z);
    // 自动距离
    const dist = maxSize * 1.5;
    const dir = new THREE.Vector3(0, 0.6, 1).normalize();
    camera.position.copy(center).add(dir.multiplyScalar(dist));
    controls.target.copy(center);
    controls.update();
}

function onWindowResize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    composer.setSize(w, h);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    composer.render();

}




