import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from aiohttp import web
from aiohttp.web_exceptions import HTTPForbidden, HTTPError

import server
import folder_paths
import comfy

wd14tagger = __import__("ComfyUI-WD14-Tagger").wd14tagger

checkpoints_list = folder_paths.get_filename_list("checkpoints")

allowed_ips = ["127.0.0.1"]
param = {"mode": "", "prompt": "", "negativePrompt": "", "seed": 0, "keepSeed": 0, "steps": 13, "cfg": 4, "interval": 30, "sampler": "dpmpp_2m,sgm_uniform", "checkpoint": "", "lora": "", "startstep": 0, "frames": 1, "framewait": 1, "offsetX": 0, "offsetY": 0, "scale": 100}
state = {"autoUpdate": False, "presetTitle": time.strftime("%Y%m%d-%H%M"), "presetFolder": "", "presetFile": "", "loraRate": "1.0", "checkpointFolder": "", "loraFolder": "", "loraFile": "", "loraTagOptions": "[]", "loraTag": "", "loraLinkHref": "", "loraPreviewSrc": "", "wd14th": 0.35, "wd14cth": 0.85}
frame_updating = False
frame_buffer = []
exclude_tags = ""


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

HEAD=r"""
<head>
<style type="text/css">
html {
    background-color: black;
}
body {
    color: lightgray;
    background: url(/stream) no-repeat top center local;
    margin: 0px;
    padding: 0px;
}
textarea {
    color: lightgray;
    background-color: black;
    width: 100%;
    font-size: 100%;
}
input {
    color: lightgray;
    background-color: black;
    font-size: 100%;
}
button {
    color: black;
    background-color: dimgray;
    font-size: 100%;
}
select {
    color: lightgray;
    background-color: black;
    font-size: 100%;
}
#captureDialog {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 999;
}
</style>
<title>FlipStreamViewer</title>
</head>
"""

SCRIPT_PARAM=r"""
function getStateAsJson() {
    const autoUpdate = document.getElementById("autoUpdateCheckbox").checked;
    const presetTitle = document.getElementById("presetTitleInput").value;
    const presetFolder = document.getElementById("presetFolderSelect").value;
    const presetFile = document.getElementById("presetFileSelect").value;
    const checkpointFolder = document.getElementById("checkpointFolderSelect").value;
    const loraFolder = document.getElementById("loraFolderSelect").value;
    const loraFile = document.getElementById("loraFileSelect").value;
    const loraTagSelect = document.getElementById("loraTagSelect");
    const loraTagOptions = JSON.stringify([...loraTagSelect.options].map(o => ({ value: o.value, text: o.text })));
    const loraTag = document.getElementById("loraTagSelect").value;
    const loraRate = document.getElementById("loraRate").value;
    const loraLinkHref = document.getElementById("loraLink").href;
    const loraPreviewSrc = document.getElementById("loraPreview").src;
    const wd14th = parseFloat(document.getElementById("wd14thRange").value);
    const wd14cth = parseFloat(document.getElementById("wd14cthRange").value);
    return { autoUpdate: autoUpdate, presetTitle: presetTitle, presetFolder: presetFolder, presetFile: presetFile, checkpointFolder: checkpointFolder, loraRate: loraRate, loraFolder: loraFolder, loraFile: loraFile, loraTagOptions: loraTagOptions, loraTag: loraTag, loraLinkHref: loraLinkHref, loraPreviewSrc: loraPreviewSrc, loraTagOptions: loraTagOptions, wd14th: wd14th, wd14cth: wd14cth };
}

function getParamAsJson() {
    const prompt = document.getElementById("promptInput").value;
    const negativePrompt = document.getElementById("negativePromptInput").value;
    const seed = parseInt(document.getElementById("seedInput").value) || 0;
    const keepSeed = parseInt(document.getElementById("keepSeedInput").value) || 0;
    const steps = parseInt(document.getElementById("stepsRange").value);
    const cfg = parseFloat(document.getElementById("cfgRange").value);
    const interval = parseInt(document.getElementById("intervalSelect").value);
    const sampler = document.getElementById("samplerSelect").value;
    const checkpoint = document.getElementById("checkpointFileSelect").value;
    const lora = document.getElementById("loraInput").value;
    const startstep =  parseInt(document.getElementById("startstepRange").value);
    const frames = parseFloat(document.getElementById("framesRange").value);
    const framewait = parseFloat(document.getElementById("framewaitRange").value);
    const offsetX = parseInt(document.getElementById("offsetXRange").value);
    const offsetY = parseInt(document.getElementById("offsetYRange").value);
    const scale = parseInt(document.getElementById("scaleRange").value);
    return { prompt: prompt, negativePrompt: negativePrompt, seed: seed, keepSeed: keepSeed, steps: steps, cfg: cfg, interval: interval, sampler: sampler, checkpoint: checkpoint, lora: lora, startstep: startstep, frames: frames, framewait: framewait, offsetX: offsetX, offsetY: offsetY, scale: scale };
}

function setParam(param) {
    document.getElementById("promptInput").value = param.prompt || "";
    document.getElementById("negativePromptInput").value = param.negativePrompt || "";
    document.getElementById("seedInput").value = param.seed || 0;
    document.getElementById("keepSeedInput").value = param.keepSeed || 0;
    document.getElementById("stepsRange").value = param.steps || 13;
    document.getElementById("stepsValue").innerText = param.steps || 13;
    document.getElementById("cfgRange").value = param.cfg || 4;
    document.getElementById("cfgValue").innerText = param.cfg || 4;
    document.getElementById("intervalSelect").value = param.interval || 30;
    document.getElementById("samplerSelect").value = param.sampler || "dpmpp_2m,sgm_uniform";
    document.getElementById("checkpointFileSelect").value = param.checkpoint || "";
    document.getElementById("loraInput").value = param.lora || "";
    document.getElementById("startstepRange").value = param.startstep || 0;
    document.getElementById("startstepValue").innerText = param.startstep || 0;
    document.getElementById("framesRange").value = param.frames || 1;
    document.getElementById("framesValue").innerText = param.frames || 1;
    document.getElementById("framewaitRange").value = param.framewait || 1;
    document.getElementById("framewaitValue").innerText = param.framewait || 1;
    document.getElementById("offsetXRange").value = param.offsetX || 0;
    document.getElementById("offsetYRange").value = param.offsetY || 0;
    document.getElementById("scaleRange").value = param.scale || 100;
}

function updateParam(reload=false) {
    var updateButton = document.getElementById("updateButton");
    updateButton.disabled = true;
    fetch("/update_param", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(), getParamAsJson()]),
        headers: {"Content-Type": "application/json"}         
    }).then(response => {
        if (response.ok) {
            if (reload) {
                location.reload();
            }
        } else {
            alert("Failed to update parameter.");
        }
    }).catch(error => {
        alert("An error occurred while updating parameter.");
    });
    updateButton.disabled = false;
}
"""

SCRIPT_PRESET=r"""
function loadPreset(promptOnly=false) {
    var presetFile = document.getElementById("presetFileSelect");
    var presetTitle = document.getElementById("presetTitleInput")
    fetch("/load_preset", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(), promptOnly]),
        headers: {"Content-Type": "application/json"}
    }).then(response => {
        if (response.ok) {
            location.reload();
        } else {
            alert("Failed to load preset.");
        }
    }).catch(error => {
        alert("An error occurred while loading preset: " + error);
    });
}

function savePreset() {
    var title = document.getElementById("presetTitleInput").value;
    if (title == "") {
        alert("Please enter a title to save.");
        return;
    }
    if (!/^[a-zA-Z0-9-_]+$/.test(title)) {
        alert("Title can only contain [a-zA-Z0-9-_] characters.");
        return;
    }
    fetch("/save_preset", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(), getParamAsJson()]),
        headers: {"Content-Type": "application/json"}
    }).then(response => {
        if (response.ok) {
            alert("Preset saved successfully!");
            location.reload();
        } else {
            alert("Failed to save preset.");
        }
    }).catch(error => {
        alert("An error occurred while saving preset.");
    });
}

function deletePreset() {
    var title = document.getElementById("presetTitleInput").value;
    if (title == "") {
        alert("Please enter a title to delete.");
        return;
    }
    if (!/^[a-zA-Z0-9-_]+$/.test(title)) {
        alert("Title can only contain [a-zA-Z0-9-_] characters.");
        return;
    }
    fetch("/delete_preset", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(), getParamAsJson()]),
        headers: {"Content-Type": "application/json"}
    }).then(response => {
        if (response.ok) {
            alert("Preset deleted successfully!");
            location.reload();
        } else {
            alert("Failed to delete preset.");
        }
    }).catch(error => {
        alert("An error occurred while deleting preset.");
    });
}
"""

SCRIPT_CKPT=r"""
function moveCheckpoint() {
    const moveCheckpointSelect = document.getElementById("moveCheckpointSelect");
    moveCheckpointSelect.style.display = "block";
}

function moveCheckpointFile() {
    const moveCheckpointSelect = document.getElementById("moveCheckpointSelect");
    const checkpointFolder = document.getElementById("checkpointFolderSelect").value;
    const checkpoint = document.getElementById("checkpointFileSelect").value;
    const checkpointFolderTo = moveCheckpointSelect.value;

    fetch("/move_checkpointfile", {
        method: "POST",
        body: JSON.stringify({checkpoint: checkpoint, checkpointFolderTo: checkpointFolderTo}),
        headers: {"Content-Type": "application/json"}
    }).then(response => {
        if (response.ok) {
            alert("Checkpoint moved successfully!");
            location.reload();
        } else {
            alert("Failed to move checkpoint.");
        }
    }).catch(error => {
        alert("An error occurred while moving checkpoint.");
    });
}
"""

SCRIPT_LORA=r"""
function selectLoraFile() {
    const loraFolder = document.getElementById("loraFolderSelect").value;
    const loraFileSelect = document.getElementById("loraFileSelect");
    const loraName = loraFileSelect.options[loraFileSelect.selectedIndex].text;
    const loraFile = loraFileSelect.value;
    const loraRate = document.getElementById("loraRate").value;
    const loraInput = document.getElementById("loraInput");
    const loraTagSelect = document.getElementById("loraTagSelect");
    const loraLink = document.getElementById("loraLink");
    const loraPreview = document.getElementById("loraPreview");
    
    if (loraFile) {
        const re = new RegExp("[\\n]?<lora:" + loraName + ":[0-9.]+>", "g");
        if (loraInput.value.search(re) == -1) {
            loraInput.value += "\n<lora:" + loraName + ":" + loraRate + ">";
        }
    }
    
    fetch("/get_lorainfo", {
        method: "POST",
        body: JSON.stringify({loraFolder: loraFolder, loraFile: loraFile}),
        headers: {"Content-Type": "application/json"}
    }).then(response => response.json()).then(json => {
        loraTagSelect.innerHTML = "";
        
        item = document.createElement("option");
        item.value = "";
        item.text = "tag";
        loraTagSelect.appendChild(item);

        if (json.ss_tag_frequency) {
            const datasets = JSON.parse(json.ss_tag_frequency);
            const tags = {};
            for (const setName in datasets) {
                const set = datasets[setName];
                for (const t in set) {
                    if (t in tags) {
                        tags[t] += set[t];
                    } else {
                        tags[t] = set[t];
                    }
                }
            }
            const sorted_tags = Object.entries(tags).sort((a, b) => b[1] - a[1]);
            for (const i in sorted_tags) {
                item = document.createElement("option");
                item.value = sorted_tags[i][0].trim();
                item.text = sorted_tags[i][0].trim() + ":" + sorted_tags[i][1];
                loraTagSelect.appendChild(item);
            }
        }
        const hash = json._lorapreview_hash;
        if (hash) {
            fetch("https://civitai.com/api/v1/model-versions/by-hash/" + hash
            ).then(response => response.json()).then(json => {
                loraLink.href = "https://civitai.com/models/" + json.modelId;
                loraPreview.src = json.images[0].url;
            }).catch(error => {
                loraPreview.src = "";
            });
        }
    });
}

function toggleLora() {
    const loraFileSelect = document.getElementById("loraFileSelect");
    const loraName = loraFileSelect.options[loraFileSelect.selectedIndex].text;
    const loraFile = loraFileSelect.value;
    const loraRate = document.getElementById("loraRate").value;
    const loraInput = document.getElementById("loraInput");
    if (loraFile) {
        const re = new RegExp("[\\n]?<lora:" + loraName + ":[0-9.]+>", "g");
        if (loraInput.value.search(re) >= 0) {
            loraInput.value = loraInput.value.replace(re, "");
        } else {
            loraInput.value += "\n<lora:" + loraName + ":" + loraRate + ">";
        }
    }
}

function moveLora() {
    const moveLoraSelect = document.getElementById("moveLoraSelect");
    moveLoraSelect.style.display = "block";
}

function moveLoraFile() {
    const moveLoraSelect = document.getElementById("moveLoraSelect");
    const loraFolder = document.getElementById("loraFolderSelect").value;
    const loraFile = document.getElementById("loraFileSelect").value;
    const loraFolderTo = moveLoraSelect.value;

    fetch("/move_lorafile", {
        method: "POST",
        body: JSON.stringify({loraFolder: loraFolder, loraFile: loraFile, loraFolderTo: loraFolderTo}),
        headers: {"Content-Type": "application/json"}
    }).then(response => {
        if (response.ok) {
            alert("Lora moved successfully!");
            location.reload();
        } else {
            alert("Failed to move lora.");
        }
    }).catch(error => {
        alert("An error occurred while moving lora.");
    });
}
"""

SCRIPT_TAG=r"""
function toggleTag() {
    const loraTag = document.getElementById("loraTagSelect").value;
    const loraInput = document.getElementById("loraInput");
    if (loraTag) {
        const re = new RegExp("^\\s*" + loraTag + "\\s*(,\\s*|\n|$)|,\\s*" + loraTag + "\\s*(,|\n|$)", "g");
        if (loraInput.value.search(re) >= 0) {
            loraInput.value = loraInput.value.replace(re, "$2")
        } else {
            if (loraInput.value) {
                loraInput.value += ", " + loraTag;
            } else {
                loraInput.value = loraTag;
            }
        }
    }
}

function randomTag() {
    const options = document.getElementById("loraTagSelect").options;
    const loraInput = document.getElementById("loraInput");
    const loraTag = options[Math.floor(Math.random() * options.length)].value;
    if (loraTag) {
        const re = new RegExp("^\\s*" + loraTag + "\\s*(,\\s*|\n|$)|,\\s*" + loraTag + "\\s*(,|\n|$)", "g");
        if (loraInput.value.search(re) < 0) {
            if (loraInput.value) {
                loraInput.value += ", " + loraTag;
            } else {
                loraInput.value = loraTag;
            }
        }
    }
}

async function addWD14Tag() {
    const loraInput = document.getElementById("loraInput");
    const wd14th = parseFloat(document.getElementById("wd14thRange").value);
    const wd14cth = parseFloat(document.getElementById("wd14cthRange").value);

    fetch("/get_wd14tag", {
        method: "POST",
        body: JSON.stringify({wd14th: wd14th, wd14cth: wd14cth}),
        headers: {"Content-Type": "application/json"}
    }).then(response => response.json()).then(json => {
        loraInput.value += "\n" + json.tags + "\n";
    }).catch(error => {
        alert(`Failed to fetch WD14 tags: ${error.stack}`);
    });
}

function clearLoraInput() {
    document.getElementById("loraInput").value = "";
}
"""

SCRIPT_SEED=r"""
function changeSeed(reload=false) {
    const seedInput = document.getElementById("seedInput");
    seedInput.value = Math.floor(Math.random() * 99999);
    updateParam(reload);
}

function keepSeed() {
    const seed = parseInt(document.getElementById("seedInput").value) || 0;
    const keepSeedInput = document.getElementById("keepSeedInput");
    keepSeedInput.value = seed;
}

function backSeed() {
    const keepSeed = parseInt(document.getElementById("keepSeedInput").value) || 0;
    const seedInput = document.getElementById("seedInput");
    seedInput.value = keepSeed;
}

var intervalId = 0;
function toggleAutoUpdate() {
    const autoUpdate = document.getElementById("autoUpdateCheckbox").checked;
    const interval = parseInt(document.getElementById("intervalSelect").value);
    if (autoUpdate) {
        intervalId = setInterval(function() {
            changeSeed();
        }, interval * 1000);
    } else {
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = 0;
        }
    }
}
toggleAutoUpdate();
"""

SCRIPT_VIEW=r"""
var toggleViewFlag = true;
function toggleView() {
    const leftPanel = document.getElementById("leftPanel");
    const rightPanel = document.getElementById("rightPanel");
    if (toggleViewFlag) {
        leftPanel.style.visibility = "hidden";
        rightPanel.style.visibility = "hidden";
        toggleViewFlag = false;
    } else {
        document.body.style.backgroundImage = "none";
        leftPanel.style.visibility = "visible";
        rightPanel.style.visibility = "visible";
        toggleViewFlag = true;
    }
}

function hideTimer() {
    setTimeout(function() {
        document.getElementById("autoUpdateCheckbox").checked = false;
        toggleAutoUpdate();
        closeCapture();
        document.body.style.backgroundImage = "none";
        leftPanel.style.visibility = "hidden";
        rightPanel.style.visibility = "hidden";
        toggleViewFlag = false;
    }, 300 * 1000);
}
hideTimer();
"""

SCRIPT_CAPTURE=r"""
let capturedCanvas = document.createElement("canvas");
async function capture() {
    try {
        const video = document.createElement("video");
        video.srcObject = await navigator.mediaDevices.getDisplayMedia({ video: true });
        video.style.display = "none";
        video.play();
        video.addEventListener("loadedmetadata", () => {
            const { videoWidth, videoHeight } = video;
            capturedCtx = capturedCanvas.getContext("2d");
            capturedCtx.clearRect(0, 0, capturedCanvas.width, capturedCanvas.height);
            capturedCanvas.width = videoWidth;
            capturedCanvas.height = videoHeight;
            capturedCtx.drawImage(video, 0, 0);
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            updateCanvas();
        });
    } catch (error) {
        alert("An error occurred while capture.");
    }
}

function updateCanvas() {
    document.getElementById("captureDialog").style.display = "flex";
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const scale = parseInt(document.getElementById("scaleRange").value, 10) / 100;
    const offsetX = parseInt(document.getElementById("offsetXRange").value, 10);
    const offsetY = parseInt(document.getElementById("offsetYRange").value, 10);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(scale, scale);
    ctx.translate(offsetX, offsetY);
    ctx.drawImage(capturedCanvas, 0, 0);
    ctx.restore();
}

function closeCapture() {
    document.getElementById("captureDialog").style.display = "none";
}

function setFrame() {
    const canvas = document.getElementById("canvas");
    const dataURL = canvas.toDataURL("image/webp");
    fetch("/set_frame", {
        method: "POST",
        body: JSON.stringify([dataURL]),
        headers: { "Content-Type": "application/json" }
    }).then(response => {
        if (response.ok) {
            closeCapture();
        } else {
            alert("Failed to set frame.");
        }
    }).catch(error => {
        alert("An error occurred while set frame.");
    });
}
"""


@server.PromptServer.instance.routes.get("/stream")
async def stream(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    response = web.StreamResponse()
    response.content_type = "multipart/x-mixed-replace; boundary=frame"
    await response.prepare(request)
    
    update_start = None
    while request.transport and not request.transport.is_closing():
        if frame_buffer:
            if len(frame_buffer) > 1:
                frame = frame_buffer.pop(0)
                frame_buffer.append(frame)
            else:
                frame = frame_buffer[0]
            
            if frame_updating:
                if update_start is None:
                    update_start = time.time()
                with BytesIO(frame) as inbuf:
                    image = Image.open(inbuf)
                    w, h = image.size
                    s = (time.time() - update_start + 1) * 4
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((int(w / 2 - s), 16, int(w / 2 + s), 32), fill="blue", outline="blue", width=1)
                    with BytesIO() as outbuf:
                        image.save(outbuf, format="WEBP", quality=100)
                        frame = outbuf.getvalue()
            else:
                update_start = None
            
            await response.write(b"--frame\r\nContent-Type: image/webp\r\n\r\n" + frame + b"\r\n")
            await response.write(b"--frame\r\nContent-Type: image/webp\r\n\r\n" + frame + b"\r\n")

        if param["framewait"] > 0:
            await asyncio.sleep(param["framewait"] * 0.1)

    return response


@server.PromptServer.instance.routes.get("/viewer")
async def viewer(request):
    if request.remote not in allowed_ips:
        print(request.remote)
        raise HTTPForbidden()

    return web.Response(text=f"""<html>{HEAD}<body>
    <div style="display: flex;">
        <div id="leftPanel" style="width: 12%;">
            <textarea id="promptInput" placeholder="Enter prompt" rows="15">{param["prompt"]}</textarea>
            <br>
            <textarea id="negativePromptInput" placeholder="Enter negativePrompt" rows="2">{param["negativePrompt"]}</textarea>
            <br>
            <button id="updateButton" onclick="updateParam(true)">Update</button>
            <button id="changeSeedButton" onclick="changeSeed(true)">Change</button>
            <br>
            Seed: <input id="seedInput" type="number" placeholder="Enter seed" value="{param["seed"]}" style="width: 5em;" />
            <br>
            <button id="keepSeedButton" onclick="keepSeed()">K</button>
            <button id="backSeedButton" onclick="backSeed()">B</button>
            <input id="keepSeedInput" type="number" placeholder="keepSeed" value="{param["keepSeed"]}" style="width: 5em;" />
            <br>
            Iv:<select id="intervalSelect" onchange="updateParam()">
                <option value="10" {"selected" if param["interval"] == 10 else ""}>10</option>
                <option value="15" {"selected" if param["interval"] == 15 else ""}>15</option>
                <option value="20" {"selected" if param["interval"] == 20 else ""}>20</option>
                <option value="25" {"selected" if param["interval"] == 25 else ""}>25</option>
                <option value="30" {"selected" if param["interval"] == 30 else ""}>30</option>
            </select>
            <input id="autoUpdateCheckbox" type="checkbox" onchange="toggleAutoUpdate()"{" checked" if state["autoUpdate"] else ""}>Auto</input>
            <br>
            <input id="stepsRange" type="range" min="1" max="25" step="1" value="{param["steps"]}" style="width: 50%;" oninput="stepsValue.innerText = this.value;" />
            <span id="stepsValue">{param["steps"]}</span>stp
            <br>
            <input id="cfgRange" type="range" min="1.0" max="7.0" step="0.1" value="{param["cfg"]}" style="width: 50%;" oninput="cfgValue.innerText = this.value;" />
            <span id="cfgValue">{param["cfg"]}</span>cfg
            <br>
            <input id="startstepRange" type="range" min="0" max="6" step="1" value="{param["startstep"]}" style="width: 50%;" oninput="startstepValue.innerText = this.value;" />
            <span id="startstepValue">{param["startstep"]}</span>sta
            <br>
            <input id="framesRange" type="range" min="1" max="8" step="1" value="{param["frames"]}" style="width: 50%;" oninput="framesValue.innerText = this.value;" />
            <span id="framesValue">{param["frames"]}</span>frm
            <br>
            <input id="framewaitRange" type="range" min="0" max="2" step="0.1" value="{param["framewait"]}" style="width: 50%;" oninput="framewaitValue.innerText = this.value;" />
            <span id="framewaitValue">{param["framewait"]}</span>spf
            <br>
            <select id="samplerSelect" onchange="updateParam()" style="width: 100%;">
                <option value="dpmpp_2m,sgm_uniform" {"selected" if param["sampler"] == "dpmpp_2m,sgm_uniform" else ""}>dpmpp_2m,sgm_uniform</option>
                <option value="ddim,ddim_uniform" {"selected" if param["sampler"] == "ddim,ddim_uniform" else ""}>ddim,ddim_uniform</option>
                <option value="eular,normal" {"selected" if param["sampler"] == "eular,normal" else ""}>eular,normal</option>
            </select>
            <br>
            <select id="checkpointFolderSelect" onchange="updateParam(true)" style="width: 100%;">
                <option value="">checkpoint folder</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["checkpointFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("ComfyUI/models/checkpoints", param["mode"]).glob("*/")])}
            </select>
            <select id="checkpointFileSelect" style="width: 50%;">
                <option value="">checkpoint</option>
                {f'<option value="{param["checkpoint"]}" selected>*{Path(param["checkpoint"]).stem}</option>' if param['checkpoint'] else ""}
                {"".join([f'<option value="{file.parent.name + "/" + file.name}">{file.stem}</option>' if param["checkpoint"] != file.parent.name + "/" + file.name else "" for file in Path("ComfyUI/models/checkpoints", param["mode"], state["checkpointFolder"]).glob("*.safetensors")])}
            </select>
            <button id="moveCheckpoint" onClick="moveCheckpoint()">M</button>
            <select id="moveCheckpointSelect" onchange="moveCheckpointFile()" style="width: 100%; display: none;">
                <option value="default" disabled selected>move to</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["checkpointFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("ComfyUI/models/checkpoints", param["mode"]).glob("*/")])}
            </select>
            <select id="presetFolderSelect" onchange="updateParam(true)" style="width: 100%;">
                <option value="">preset folder</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["presetFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("preset").glob("*/")])}
            </select>
            <select id="presetFileSelect" style="width: 50%;">
                <option value="" disabled selected>preset</option>
                {"".join([f'<option value="{file.name}"{" selected" if state["presetFile"] == file.name else ""}>{file.stem}</option>' for file in Path("preset", state["presetFolder"]).glob("*.json")])}
            </select>
            <button id="loadPreset" onclick="loadPreset()">L</button>
            <button id="loadPresetPrompt" onclick="loadPreset(true)">P</button>
            <br>
            <input id="presetTitleInput" placeholder="preset title" value="{state["presetTitle"]}" style="width: 50%;" />
            <button id="savePreset" onclick="savePreset()">S</button>
            <button id="deletePreset" onclick="deletePreset()">D</button>
        </div>
        <div id="centerPanel" style="width: 76%;">
            <button id="toggleView" onclick="toggleView()" style="width: 100%; height: 100%; border: none; background: transparent;"></button>
        </div>
        <div id="rightPanel" style="width: 12%;">
            <select id="loraFolderSelect" onchange="updateParam(true)" style="width: 100%;">
                <option value="">lora folder</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["loraFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("ComfyUI/models/loras", param["mode"]).glob("*/")])}
            </select>
            <select id="loraFileSelect" onchange="selectLoraFile()" style="width: 50%;">
                <option value="" disabled selected>lora file</option>
                {"".join([f'<option value="{file.name}"{" selected" if state["loraFile"] == file.name else ""}>{file.stem}</option>' for file in Path("ComfyUI/models/loras", param["mode"], state["loraFolder"]).glob("*.safetensors")])}
            </select>
            <button id="toggleLora" onclick="toggleLora()">T</button>
            <button id="moveLora" onClick="moveLora()">M</button>
            <br>
            <select id="moveLoraSelect" onchange="moveLoraFile()" style="width: 100%; display: none;">
                <option value="default" disabled selected>move to</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["loraFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("ComfyUI/models/loras", param["mode"]).glob("*/")])}
            </select>
            <select id="loraTagSelect" onchange="toggleTag()" style="width: 50%;">
                <option value="" disabled selected>tag</option>
                {"".join([f'<option value="{opt["value"]}"{" selected" if state["loraTag"] == opt["value"] else ""}>{opt["text"]}</option>' if opt["value"] else "" for opt in json.loads(state["loraTagOptions"])])}
            </select>
            <button id="toggleTag" onclick="toggleTag()">T</button>
            <button id="randomTag" onclick="randomTag()">R</button>
            <br>
            <button id="updateButton2" onclick="updateParam(true)">U</button>
            <button id="wd14Button" onclick="addWD14Tag()">W</button>
            <button id="captureButton" onclick="capture()">C</button>
            <select id="loraRate">
                <option value="1" {"selected" if state["loraRate"] == "1" else ""}>1</option>
                <option value="0.9" {"selected" if state["loraRate"] == "0.9" else ""}>0.9</option>
                <option value="0.8" {"selected" if state["loraRate"] == "0.8" else ""}>0.8</option>
                <option value="0.7" {"selected" if state["loraRate"] == "0.7" else ""}>0.7</option>
                <option value="0.6" {"selected" if state["loraRate"] == "0.6" else ""}>0.6</option>
                <option value="0.5" {"selected" if state["loraRate"] == "0.5" else ""}>0.5</option>
                <option value="0.4" {"selected" if state["loraRate"] == "0.4" else ""}>0.4</option>
                <option value="0.3" {"selected" if state["loraRate"] == "0.3" else ""}>0.3</option>
                <option value="0.2" {"selected" if state["loraRate"] == "0.2" else ""}>0.2</option>
                <option value="0.1" {"selected" if state["loraRate"] == "0.1" else ""}>0.1</option>
            </select>
            <br>
            <input id="wd14thRange" type="range" min="0" max="1" step="0.01" value="{state["wd14th"]}" style="width: 50%;" oninput="wd14thValue.innerText = this.value;" />
            <span id="wd14thValue">{state["wd14th"]}</span>wth
            <br>
            <input id="wd14cthRange" type="range" min="0" max="1" step="0.01" value="{state["wd14cth"]}" style="width: 50%;" oninput="wd14cthValue.innerText = this.value;" />
            <span id="wd14cthValue">{state["wd14cth"]}</span>cth
            <br>
            <button id="clearLoraInputButton" onclick="clearLoraInput()">Clear Below</button>
            <br>
            <textarea id="loraInput" placeholder="Enter lora" rows="15">{param["lora"]}</textarea>
            <br>
            <a id="loraLink" href="{state["loraLinkHref"]}" target="_blank">
                <img id="loraPreview" src="{state["loraPreviewSrc"]}" style="max-width: 100%; max-height: 15em; width:auto; height:auto; margin:auto; display:block;" />
            </a>
        </div>
    </div>
    <div id="captureDialog">
        <div style="width: 12%;"></div>
        <div style="width: 76%; text-align: center;">
            <canvas id="canvas" width="960" height="600"/>
        </div>
        <div style="width: 12%;">
            X <input id="offsetXRange" type="range" min="-960" max="960" value="{param["offsetX"]}" onchange="updateCanvas()" style="width: 80%;">
            <br>
            Y <input id="offsetYRange" type="range" min="-600" max="600" value="{param["offsetY"]}" onchange="updateCanvas()" style="width: 80%;">
            <br>
            S <input id="scaleRange" type="range" min="50" max="400" value="{param["scale"]}" onchange="updateCanvas()" style="width: 80%;">
            <br>
            <button id="setframeButton" onclick="setFrame()">SetFrame</button>
            <button id="closeButton" onclick="closeCapture()">Close</button>
        </div>
    </div>
    <script>
    {SCRIPT_PARAM}
    {SCRIPT_PRESET}
    {SCRIPT_CKPT}
    {SCRIPT_LORA}
    {SCRIPT_TAG}
    {SCRIPT_SEED}
    {SCRIPT_VIEW}
    {SCRIPT_CAPTURE}
    </script>
    </body></html>""", content_type="text/html")


@server.PromptServer.instance.routes.post("/update_param")
async def update_param(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, prm = await request.json()
    state.update(stt)
    param.update(prm)
    return web.Response(status=200)


@server.PromptServer.instance.routes.post("/get_lorainfo")
async def get_lorainfo(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    req = await request.json()
    loraPath = Path("ComfyUI/models/loras", param["mode"], req["loraFolder"], req["loraFile"])
    hash = None
    with open(loraPath, "rb") as file:
        header_size = int.from_bytes(file.read(8), "little", signed=False)
        if header_size <= 0:
            raise HTTPError("get_lorainfo: Invalid header size")

        header = file.read(header_size)
        if header_size <= 0:
            raise HTTPError("get_lorainfo: Invalid header")
        
        file.seek(0)
        hash = hashlib.sha256(file.read()).hexdigest()

    header_json = json.loads(header)
    lorainfo = header_json["__metadata__"] if "__metadata__" in header_json else {}
    lorainfo["_lorapreview_hash"] = hash
    return web.json_response(lorainfo)


@server.PromptServer.instance.routes.post("/get_wd14tag")
async def get_wd14tag(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()
    
    prm = await request.json()
    tags = []
    if frame_buffer:
        with BytesIO(frame_buffer[0]) as buf:
            image = Image.open(buf)
            tags = await wd14tagger.tag(image, "wd-v1-4-moat-tagger-v2.onnx", prm["wd14th"], prm["wd14cth"], exclude_tags)
    return web.json_response({"tags": tags})


@server.PromptServer.instance.routes.post("/set_frame")
async def set_frame(request):
    global frame_buffer
    if request.remote not in allowed_ips:
        raise HTTPForbidden()
    
    dataURL, = await request.json()
    _, data = dataURL.split(",", 1)
    frame_buffer = [base64.b64decode(data)]
    return web.Response(status=200)


@server.PromptServer.instance.routes.post("/move_checkpointfile")
async def move_lorafile(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    req = await request.json()
    loraPathFrom = Path("ComfyUI/models/checkpoints", param["mode"], req["checkpoint"])
    loraPathTo = Path("ComfyUI/models/checkpoints", param["mode"], req["checkpointFolderTo"], loraPathFrom.name)
    Path(loraPathFrom).rename(loraPathTo)
    state["checkpointFolder"] = req["checkpointFolderTo"]
    return web.Response(status=200)


@server.PromptServer.instance.routes.post("/move_lorafile")
async def move_lorafile(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    req = await request.json()
    loraPathFrom = Path("ComfyUI/models/loras", param["mode"], req["loraFolder"], req["loraFile"])
    loraPathTo = Path("ComfyUI/models/loras", param["mode"], req["loraFolderTo"], req["loraFile"])
    Path(loraPathFrom).rename(loraPathTo)
    state["loraFolder"] = req["loraFolderTo"]
    state["loraFile"] = req["loraFile"]
    return web.Response(status=200)


@server.PromptServer.instance.routes.post("/load_preset")
async def load_preset(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, promptOnly = await request.json()
    state.update(stt)
    data = param
    with open(Path("preset", state["presetFolder"], state["presetFile"]), "r") as file:
        buf = json.load(file)
        if promptOnly:
            data["prompt"] = buf["prompt"]
            if "postfix" in buf:
                data["postfix"] = buf["postfix"]
            if "negativePrompt" in buf:
                data["negativePrompt"] = buf["negativePrompt"]
        else:
            data = buf

    data["prompt"] = base64.b64decode(data["prompt"]).decode("utf-8")
    if "postfix" in buf:
        data["prompt"] += base64.b64decode(data["postfix"]).decode("utf-8")
    data["negativePrompt"] = base64.b64decode(data["negativePrompt"]).decode("utf-8") if "negativePrompt" in data else ""
    try:
        data["lora"] = base64.b64decode(data["lora"]).decode("utf-8")
    except Exception as ex:
        print(ex)

    param.update(data)
    state["presetTitle"] = Path(state["presetFile"]).stem
    return web.Response(status=200)


@server.PromptServer.instance.routes.post("/save_preset")
async def save_preset(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    [stt, data] = await request.json()
    state.update(stt)
    param.update(data)
    data["prompt"] = base64.b64encode(bytes(data["prompt"], "utf-8")).decode("utf-8")
    data["negativePrompt"] = base64.b64encode(bytes(data["negativePrompt"], "utf-8")).decode("utf-8")
    data["lora"] = base64.b64encode(bytes(data["lora"], "utf-8")).decode("utf-8")
    with open(Path("preset", state["presetFolder"], state["presetTitle"] + ".json"), "w") as file:
        json.dump(data, file)
    return web.Response(status=200)


@server.PromptServer.instance.routes.post("/delete_preset")
async def delete_preset(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    [stt, data] = await request.json()
    state.update(stt)
    param.update(data)
    Path("preset", state["presetFolder"], state["presetTitle"] + ".json").unlink(missing_ok=True)
    return web.Response(status=200)


class FlipStreamLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": ("STRING", {"default": param["mode"]}),
                "default_ckpt": (checkpoints_list,),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "loader"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return param["checkpoint"]
    
    def loader(self, mode, default_ckpt):
        global frame_updating
        frame_updating = True
        param["mode"] = mode
        if param["checkpoint"]:
            ckpt_path = folder_paths.get_full_path("checkpoints", param["mode"] + "/" + param["checkpoint"])
        else:
            ckpt_path = folder_paths.get_full_path("checkpoints", default_ckpt)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]


class FlipStreamUpdate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","INT","INT","FLOAT",comfy.samplers.KSampler.SAMPLERS,comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("prompt","batchPrompt","appPrompt","negativePrompt","seed","steps","cfg","sampler","scheduler")
    FUNCTION = "update"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return (param["prompt"], param["negativePrompt"], param["seed"], param["steps"], param["cfg"], param["lora"], param["sampler"])
        
    def update(self, **kwargs):
        global frame_updating
        frame_updating = True

        buf = param["prompt"].split("----\n")
        prompt = buf[0].replace("{lora}", param["lora"])
        batchPrompt = buf[1] if len(buf) > 1 else "-\n-\n"
        appPrompt = buf[2] if len(buf) > 2 else ""
        batchPrompt = ",\n".join([f'"{n}":"{item.lstrip("-").strip()}"' for n, item in enumerate(batchPrompt.strip().split("\n"))])
        sampler, scheduler = param["sampler"].split(",")
        return (prompt, batchPrompt, appPrompt, param["negativePrompt"], param["seed"], param["steps"], param["cfg"], sampler, scheduler)


class FlipStreamOption:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    
    RETURN_TYPES = ("INT","INT",["enable", "disable"])
    RETURN_NAMES = ("startstep","frames","start_noise")
    FUNCTION = "option"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return (param["startstep"], param["frames"])
    
    def option(self, **kwargs):
        if param["startstep"] == 0:
            start_noise = "enable"
        else:
            start_noise = "disable"
        return (param["startstep"], param["frames"], start_noise)
    

class FlipStreamViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("IMAGE",),
                "allowip": ("STRING", {"default": ""}),
                "wd14exc": ("STRING", {"default": ""}),
                "idle": ("FLOAT", {"default": 1.0}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, allowip, wd14exc, idle, **kwargs):
        global allowed_ips
        global exclude_tags
        allowed_ips = ["127.0.0.1"] + list(map(str.strip, allowip.split(",")))
        exclude_tags = wd14exc
        time.sleep(idle)
        return None

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "update_frame"
    CATEGORY = "FlipStreamViewer"

    def update_frame(self, tensor, **kwargs):
        global frame_updating
        global frame_buffer
        buffer = []
        for i in range(tensor.shape[0]):
            image = Image.fromarray(np.clip(255. * tensor[i,:,:,:].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            with BytesIO() as image_bytes:
                image.save(image_bytes, format="WEBP", quality=100)
                buffer.append(image_bytes.getvalue())
        buffer += buffer[::-1]
        frame_updating = False
        frame_buffer = buffer
        return ()


NODE_CLASS_MAPPINGS = {
    "FlipStreamLoader": FlipStreamLoader,
    "FlipStreamUpdate": FlipStreamUpdate,
    "FlipStreamOption": FlipStreamOption,
    "FlipStreamViewer": FlipStreamViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlipStreamLoader": "FlipStreamLoader",
    "FlipStreamUpdate": "FlipStreamUpdate",
    "FlipStreamOption": "FlipStreamOption",
    "FlipStreamViewer": "FlipStreamViewer",
}
