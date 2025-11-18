import base64
import hashlib
import html
import io
import itertools
import json
import threading
import time
from pathlib import Path

import imageio.v3 as iio
import requests
import torch
import torch.nn.functional as NNF
import transformers
import numpy as np
import mss
from PIL import Image, ImageDraw
from aiohttp import web
from aiohttp.web_exceptions import HTTPForbidden, HTTPError

import server
import folder_paths
import comfy
from nodes import CheckpointLoaderSimple, VAELoader

import sys
sys.path.append("ComfyUI/custom_nodes")

try:
    from llama_cpp import Llama  # llama-cpp-python
except:
    Llama = None

try:
    rembg = __import__("comfyui-inspyrenet-rembg")
except:
    rembg = None

try:
    florence2 = __import__("comfyui-florence2")
except:
    florence2 = None

try:
    film = __import__("comfyui-frame-interpolation.vfi_models.film").vfi_models.film
except:
    film = None

try:
    from comfyui_tensorrt import TensorRTLoader
except:
    TensorRTLoader = None

def btoa_utf8(value):
    return base64.b64encode(value.encode("utf-8")).decode()

def atob_utf8(value):
    if not value:
        return ""
    return base64.b64decode(value.encode()).decode("utf-8")

STREAM_COMPRESSION = 1
UPDATE_DELAY = 1.0
allowed_ips = ["127.0.0.1"]
refresh_data = {}
refresh_param = {}
default_param = {"lora": "", "_capture_offsetX": 0, "_capture_offsetY": 0, "_capture_scale": 100}
param = default_param.copy()
state = {"presetTitle": time.strftime("%Y%m%d-%H%M"), "presetFolder": "", "presetFile": "", "loraRate": "1", "loraRank": "0", "loraMode": "", "loraFolder": "", "loraFile": "", "loraTagOptions": "[]", "loraTag": "", "loraLinkHref": "", "loraPreviewSrc": "", "darker": 0.0}
frame_updating = None
frame_buffer = []
frame_mtime = 0
frame_fps = 16
setframe_mtime = 0
setframe_buffer = []

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
    margin: 4px;
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
    border-radius: 4px;
    padding: 0px;
    min-width: 2em;
}

button.tags {
    padding: 2px;
    min-width: 2em;
}

hr {
    background-color: dimgray;
    border: none;
    height: 3px;
    padding: 0px;
    margin: 1px;    
}

button.disabled {
    opacity: 0.5;
}

select {
    color: lightgray;
    background-color: black;
    font-size: 100%;
}

div.row {
    display: flex;
    position: relative;
}

div#presetXorkeyInputDiv {
    display: none;
    position: relative;
}

.FlipStreamSlider {
    width: 50%;
}

.FlipStreamInputBox {
    width: 100%;
}

.FlipStreamSelectBox,
.FlipStreamFolderSelect,
.FlipStreamFileSelect,
.FlipStreamMoveFileSelect {
    width: 100%;
}

.FlipStreamMoveFileSelect {
    display: none;
}

.FlipStreamPreviewBox {
    max-width: 100%;
}

.FlipStreamPreviewRoi {
    position: absolute;
    top: 0;
    left: 0;
}

#leftPanel,
#rightPanel,
#captureLeftPanel,
#captureRightPanel,
#tagLeftPanel,
#tagRightPanel,
#presetLeftPanel,
#presetRightPanel {
    width: 12%;
    background: rgba(0, 0, 0, 0.8);
}

#centerPanel,
#captureCenterPanel,
#tagCenterPanel,
#presetCenterPanel {
    width: 76%;
    text-align: center;
}

#captureCenterPanel,
#tagCenterPanel {
    background: rgba(0, 0, 0, 0.8);
}

#mainDialog, #captureDialog, #tagDialog, #presetDialog {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}

#mainDialog {
    display: flex;
    background-position: center;
    background-repeat: no-repeat;
    background-blend-mode: overlay;
}

#captureDialog, #tagDialog, #presetDialog {
    display: none;
    z-index: 999;
}

#presetFolderSelect,
#movePresetSelect,
#loraFolderSelect,
#moveLoraSelect {
    width: 100%;
}

#movePresetSelect,
#moveLoraSelect {
    display: none;
}

#presetXorkey {
    display: none;
}

#presetFileSelect,
#presetTitleInput,
#loraFileSelect,
#loraTagSelect {
    width: 80%;
}

#darkerRange {
    width: 50%;
}

#messageBox {
    width: 100%;
    height: 100%;
    border: none;
    padding: 10px;
    background: transparent;
    text-align: left;
    user-select: none;
    font-size: 1rem;
    text-shadow: 
    black 2px 0px,  black -2px 0px,
    black 0px -2px, black 0px 2px,
    black 2px 2px , black -2px 2px,
    black 2px -2px, black -2px -2px,
    black 1px 2px,  black -1px 2px,
    black 1px -2px, black -1px -2px,
    black 2px 1px,  black -2px 1px,
    black 2px -1px, black -2px -1px;
}

#loraLink img {
    max-width: 100%;
    max-height: 8em;
    width: auto;
    height: auto;
    margin: auto;
    display: block;
}

#offsetXRange,
#offsetYRange,
#scaleRange {
    width: 80%;
}

</style>
<title>FlipStreamViewer</title>
</head>
"""

SCRIPT_PARAM=r"""
function btoa_utf8(str) {
    return btoa(unescape(encodeURIComponent(str)));
}

function atob_utf8(str) {
    if (!str) {
        return "";
    }
    return decodeURIComponent(escape(atob(str)));
}

function getStateAsJson(force_state={}) {
    const presetTitle = document.getElementById("presetTitleInput").value;
    const presetFolder = document.getElementById("presetFolderSelect").value;
    const presetFile = document.getElementById("presetFileSelect").value;
    const loraFolder = document.getElementById("loraFolderSelect").value;
    const loraFile = document.getElementById("loraFileSelect").value;
    const loraTagSelect = document.getElementById("loraTagSelect");
    const loraTagOptions = JSON.stringify([...loraTagSelect.options].map(o => ({ value: o.value, text: o.text })));
    const loraTag = document.getElementById("loraTagSelect").value;
    const loraRate = document.getElementById("loraRate").value;
    const loraRank = document.getElementById("loraRank").value;
    const loraLinkHref = document.getElementById("loraLink").getAttribute("href");
    const loraPreviewSrc = document.getElementById("loraPreview").getAttribute("src");
    const darker = parseFloat(document.getElementById("darkerRange").value);
    var res = { presetTitle: presetTitle, presetFolder: presetFolder, presetFile: presetFile, loraRate: loraRate, loraRank: loraRank, loraFolder: loraFolder, loraFile: loraFile, loraTagOptions: loraTagOptions, loraTag: loraTag, loraLinkHref: loraLinkHref, loraPreviewSrc: loraPreviewSrc, loraTagOptions: loraTagOptions, darker: darker };
    document.querySelectorAll('.FlipStreamFolderSelect').forEach(x => res[x.name] = x.value);
    res = Object.assign(res, force_state);
    return res;
}

function getParamAsJson(force_param={}) {
    const lora = btoa_utf8(document.getElementById("loraInput").value.trim());
    const _capture_offsetX = parseInt(document.getElementById("offsetXRange").value);
    const _capture_offsetY = parseInt(document.getElementById("offsetYRange").value);
    const _capture_scale = parseInt(document.getElementById("scaleRange").value);

    var res = { lora: lora, _capture_offsetX: _capture_offsetX, _capture_offsetY: _capture_offsetY, _capture_scale: _capture_scale }
    document.querySelectorAll('.FlipStreamSlider').forEach(x => res[x.name] = x.value);
    document.querySelectorAll('.FlipStreamTextBox').forEach(x => res[x.name] = btoa_utf8(x.value));
    document.querySelectorAll('.FlipStreamInputBox').forEach(x => res[x.name] = x.value);
    document.querySelectorAll('.FlipStreamSelectBox').forEach(x => res[x.name] = x.value);
    document.querySelectorAll('.FlipStreamFileSelect').forEach(x => res[x.name] = x.value);
    res = Object.assign(res, force_param);
    return res;
}

function setParam(param) {
    document.getElementById("loraInput").value = param.lora || "";
    document.getElementById("offsetXRange").value = param.offsetX || 0;
    document.getElementById("offsetYRange").value = param.offsetY || 0;
    document.getElementById("scaleRange").value = param.scale || 100;
    for (let key in param) {
        const elem = document.querySelector(`input[name=${key}]`);
        elem.value = param[key] || elem.value;
    }
}

function updateParam(reload=false, force_state={}, force_param={}, search="") {
    fetch("/flipstreamviewer/update_param", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(force_state), getParamAsJson(force_param)]),
        headers: {"Content-Type": "application/json"}         
    }).then(response => {
        if (response.ok) {
            if (reload) {
                reloadPage(search);
            }
        } else {
            alert("Failed to update parameter.");
        }
    }).catch(error => {
        alert("An error occurred while updating parameter.");
    });
}

function setupPreviewRoi() {
    document.querySelectorAll('.FlipStreamPreviewRoi').forEach(canvas => {
        const ctx = canvas.getContext('2d');
        const img = canvas.parentElement.querySelector('img');
        const label = canvas.id.replace("PreviewRoi", "");
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
        };

        let drawing = false, x, y;
        canvas.onmousedown = e => { drawing = true; [x, y] = [e.offsetX, e.offsetY]; };
        canvas.onmousemove = e => { if (drawing) { canvas.width = canvas.width; ctx.strokeStyle = 'red'; ctx.strokeRect(x, y, e.offsetX - x, e.offsetY - y); } };
        canvas.onmouseup = e => {
            if (!drawing) return;
            drawing = false;
            var sx = x / canvas.width;
            var sy = y / canvas.height;
            var ex = e.offsetX / canvas.width;
            var ey = e.offsetY / canvas.height;
            if (ex <= sx || ey <= sy) {
                sx = 0;
                sy = 0;
                ex = 1;
                ey = 1;
            }
            fetch('/flipstreamviewer/preview_setroi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    label: label,
                    sx: sx,
                    sy: sy,
                    ex: ex,
                    ey: ey
                })
            });
        };
    });
}
setupPreviewRoi();
"""

SCRIPT_PRESET=r"""
function loadPreset(loraPromptOnly=false, force_state={}, search="") {
    const xorkey = document.getElementById("presetXorkeyInput").value;
    fetch("/flipstreamviewer/load_preset", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(force_state), loraPromptOnly, xorkey]),
        headers: {"Content-Type": "application/json"}
    }).then(response => response.json()).then(json => {
        if (loraPromptOnly) {
            document.getElementById("loraInput").value = atob_utf8(json.lora);
            document.getElementById("presetTitleInput").value = json.presetTitle;
        }
        else {
            reloadPage(search);
        }
    }).catch(error => {
        alert("An error occurred while loading preset: " + error);
    });
}

function movePreset() {
    document.getElementById("movePresetSelect").style.display = "block";
    document.getElementById("movePresetSelect").value = "";
}

function movePresetFile() {
    const moveFile = document.getElementById("presetFileSelect").value;
    const moveTo = document.getElementById("movePresetSelect").value;
    if (!moveFile || !moveTo) {
        return;
    }
    fetch("/flipstreamviewer/move_presetfile", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(), moveTo]),
        headers: {"Content-Type": "application/json"}
    }).then(response => {
        if (response.ok) {
            alert("Preset moved successfully!");
        } else {
            alert("Failed to move preset.");
        }
    }).catch(error => {
        alert("An error occurred while moving preset.");
    });
}

function savePreset() {
    const title = document.getElementById("presetTitleInput").value;
    const xorkey = document.getElementById("presetXorkeyInput").value;
    if (title == "") {
        alert("Please enter a title to save.");
        return;
    }
    if (!/^[a-zA-Z0-9-_]+$/.test(title)) {
        alert("Title can only contain [a-zA-Z0-9-_] characters.");
        return;
    }
    fetch("/flipstreamviewer/save_preset", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(), getParamAsJson(), xorkey]),
        headers: {"Content-Type": "application/json"}
    }).then(response => {
        if (response.ok) {
            alert("Preset saved successfully!");
        } else {
            alert("Failed to save preset.");
        }
    }).catch(error => {
        alert("An error occurred while saving preset.");
    });
}

function showPresetDialog() {
    document.getElementById("mainDialog").style.display = "none";
    document.getElementById("presetDialog").style.display = "flex";
}

function closePresetDialog() {
    document.getElementById("mainDialog").style.display = "flex";
    document.getElementById("presetDialog").style.display = "none";
}
"""

SCRIPT_CKPT=r"""
function showMoveFileSelect(obj) {
    obj.style.display="block";
    obj.value="";
}

function moveFile(folder_path, mode, label) {
    const moveFrom = document.getElementById(label + "FileSelect").value;
    const moveTo = document.getElementById(label + "MoveFileSelect").value;
    if (!moveFrom || !moveTo) {
        return;
    }
    fetch("/flipstreamviewer/move_file", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(), folder_path, mode, label, moveFrom, moveTo]),
        headers: {"Content-Type": "application/json"}
    }).then(response => {
        if (response.ok) {
            alert("Moved successfully!");
        } else {
            alert("Failed to move.");
        }
        reloadPage();
    }).catch(error => {
        alert("An error occurred while moving.");
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
    const loraTagSelect = document.getElementById("loraTagSelect");
    const loraInput = document.getElementById("loraInput");
    const loraLink = document.getElementById("loraLink");
    const loraPreview = document.getElementById("loraPreview");
    
    if (loraFile) {
        const re = new RegExp("[\\n]?<lora:" + loraName + ":[-]?[0-9.]+>", "g");
        if (loraInput.value.search(re) == -1) {
            if (loraInput.value.trim() != "") {
                loraInput.value += "\n";
            }
            loraInput.value += "<lora:" + loraName + ":" + loraRate + ">";
        }
    }
    
    fetch("/flipstreamviewer/get_lorainfo", {
        method: "POST",
        body: JSON.stringify({loraFolder: loraFolder, loraFile: loraFile}),
        headers: {"Content-Type": "application/json"}
    }).then(response => response.json()).then(json => {
        loraTagSelect.innerHTML = "";
        
        item = document.createElement("option");
        item.value = "";
        item.text = "tags";
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

        loraLink.href = json._lorapreview_href || "";
        loraPreview.src = json._lorapreview_src || "";
    });
}

function toggleLora() {
    const loraFileSelect = document.getElementById("loraFileSelect");
    const loraName = loraFileSelect.options[loraFileSelect.selectedIndex].text;
    const loraFile = loraFileSelect.value;
    const loraRate = document.getElementById("loraRate").value;
    const loraInput = document.getElementById("loraInput");
    if (loraFile) {
        const re = new RegExp("[\\n]?<lora:" + loraName + ":[-]?[0-9.]+>", "g");
        if (loraInput.value.search(re) >= 0) {
            loraInput.value = loraInput.value.replace(re, "");
        } else {
            loraInput.value += "\n<lora:" + loraName + ":" + loraRate + ">";
        }
    }
}

function moveLora() {
    document.getElementById("moveLoraSelect").style.display = "block";
    document.getElementById("moveLoraSelect").value = "";
}

function moveLoraFile() {
    const moveFile = document.getElementById("loraFileSelect").value;
    const moveTo = document.getElementById("moveLoraSelect").value;
    if (!moveFile || !moveTo) {
        return;
    }
    fetch("/flipstreamviewer/move_lorafile", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(), moveTo]),
        headers: {"Content-Type": "application/json"}
    }).then(response => {
        if (response.ok) {
            alert("Lora moved successfully!");
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
    const loraTagSelect = document.getElementById("loraTagSelect");
    const loraTag = document.getElementById("loraTagSelect").value;
    const loraRank = parseInt(document.getElementById("loraRank").value);
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
    } else {
        for (var i = 1; i < Math.min(loraRank + 1, loraTagSelect.length); i++) {
            loraInput.value += ", " + loraTagSelect.options[i].value;
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

function clearLoraInput() {
    document.getElementById("loraInput").value = "";
}

function showTagDialog() {
    const lora = document.getElementById("loraInput").value;
    const tags = lora.replace(/\n/g, ',').split(',').map(value => value.trim());
    const tagCenterPanel = document.getElementById("tagCenterPanel");
    const count = tagCenterPanel.children.length;

    for (const child of Array.from(tagCenterPanel.children)) {
        if (!child.classList.contains("disabled")) {
            child.classList.add("disabled");
        }
    }

    for (let i = 0; i < tags.length; i++) {
        if (!tags[i])
            continue;

        const child = Array.from(tagCenterPanel.children).find(child => child.value == tags[i]);
        if (child)
        {
            child.classList.remove("disabled");
            continue;
        }

        if (tags[i] == "----") {
            if (count == 0) {
                const hr = document.createElement("hr");
                hr.classList.add("tags");
                hr.classList.add("separator");
                tagCenterPanel.appendChild(hr);
            }
        } else {
            const button = document.createElement("button");
            button.value = tags[i];
            button.textContent = tags[i];
            button.classList.add("tags");
            button.addEventListener("click", function() {
                if (button.classList.contains("disabled")) {
                    button.classList.remove("disabled");
                } else {
                    button.classList.add("disabled");
                }
            });
            tagCenterPanel.appendChild(button);
        }
    }

    document.getElementById("tagDialog").style.display = "flex";
}

function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1) + min);
}

function tagCSel() {
    const tagCenterPanel = document.getElementById("tagCenterPanel");
    Array.from(tagCenterPanel.children).forEach(child => child.classList.contains("disabled") || child.classList.add("disabled"));
}

function tagRSel() {
    const tagCenterPanel = document.getElementById("tagCenterPanel");
    var group = [];
    Array.from(tagCenterPanel.children).forEach(child => {
        if (child.classList.contains("separator") && group.length > 0) {
            const r = getRandomInt(0, group.length);
            if (r != group.length) {
                group[r].classList.remove("disabled");
            }
            group = [];
        } else {
            group.push(child);
        }
    });
    const r = getRandomInt(0, group.length);
    if (r != group.length) {
        group[r].classList.remove("disabled");
    }
}

function tagOK() {
    const tagCenterPanel = document.getElementById("tagCenterPanel");
    var selectedTags = [];
    Array.from(tagCenterPanel.children).forEach(child => {
        if (!child.classList.contains("separator") && !child.classList.contains("disabled")) {
            selectedTags.push(child.value);
        }
    });
    loraInput.value = selectedTags.join(", ");
    closeTagDialog();
}

function closeTagDialog() {
    document.getElementById("tagDialog").style.display = "none";
}
"""

SCRIPT_VIEW=r"""
var toggleViewFlag = true;
var streamViewFlag = true;
var streamInfo = { mtime: 0, fps: 8, count: 0 };
var streamMTime = 0;
var streamCache = [];
var streamIndex = 0;

function detailsState() {
    const ds = document.querySelectorAll('details');
    ds.forEach((d, i) => d.open = sessionStorage.getItem(`details_state-${i}`) === 'open');
    document.addEventListener('toggle', e => {
        sessionStorage.setItem(`details_state-${Array.from(ds).indexOf(e.target)}`, e.target.open ? 'open' : 'closed');
    }, true);
}
detailsState();

function toggleView() {
    const leftPanel = document.getElementById("leftPanel");
    const rightPanel = document.getElementById("rightPanel");
    const presetLeftPanel = document.getElementById("presetLeftPanel");
    const presetRightPanel = document.getElementById("presetRightPanel");
    const messageBox = document.getElementById("messageBox");
    if (toggleViewFlag) {
        leftPanel.style.visibility = "hidden";
        rightPanel.style.visibility = "hidden";
        presetLeftPanel.style.visibility = "hidden";
        presetRightPanel.style.visibility = "hidden";
        messageBox.style.visibility = "hidden";
        toggleViewFlag = false;
    } else {
        leftPanel.style.visibility = "visible";
        rightPanel.style.visibility = "visible";
        presetLeftPanel.style.visibility = "visible";
        presetRightPanel.style.visibility = "visible";
        messageBox.style.visibility = "visible";
        toggleViewFlag = true;
    }
}

function hideView() {
    const leftPanel = document.getElementById("leftPanel");
    const centerPanel = document.getElementById("centerPanel");
    const rightPanel = document.getElementById("rightPanel");
    const presetLeftPanel = document.getElementById("presetLeftPanel");
    const presetRightPanel = document.getElementById("presetRightPanel");
    const messageBox = document.getElementById("messageBox");
    closeCaptureDialog();
    messageBox.style.visibility = "hidden";
    document.getElementById("loraPreview").src = "";
    document.querySelectorAll('.FlipStreamPreviewBox').forEach(x => x.src = "");
    leftPanel.style.visibility = "hidden";
    centerPanel.style.visibility = "hidden";
    rightPanel.style.visibility = "hidden";
    presetLeftPanel.style.visibility = "hidden";
    presetRightPanel.style.visibility = "hidden";
    toggleViewFlag = false;
    streamViewFlag = false;
}
setTimeout(hideView, 300 * 1000);

async function updateStreamView() {
    const container = document.getElementById('mainDialog');
    if (streamViewFlag && streamCache.length > 0) {
        const img = streamCache[streamIndex];
        container.style.backgroundImage = `url(${img.src})`;
        streamIndex = (streamIndex + 1) % streamCache.length;
    } else {
        container.style.backgroundImage = 'none';
    }
}
var updateStreamInterval = setInterval(updateStreamView, 1000 / streamInfo.fps);

async function refreshView() {
    const data = await fetch("/flipstreamviewer/refresh_view")
        .then(r => r.json()).catch(e => ({ status: "Fails to refresh view: " + e }));
    document.getElementById("statusInfo").textContent = data.status_info || "Empty";
    document.getElementById("messageBox").innerText = atob_utf8(data.message) || "";
    document.getElementById("messageBox").style.fontSize = data.message_fontsize || "1rem";
    for (const k in data.param) {
        const el = document.querySelector(`[name="${k}"]`);
        if (el) {
            if (el.classList.contains('FlipStreamTextBox')) {
                el.value = atob_utf8(data.param[k]);
            } else if (el.type === 'range') {
                el.value = parseFloat(data.param[k]);
            } else {
                el.value = data.param[k];
            }
        }
    }
    document.querySelectorAll('.FlipStreamLogBox').forEach(async x => {
        if (x.id in data.log) {
            x.innerText = atob_utf8(data.log[x.id]) || "";
        } else {
            x.innerText = "";
        }
    });
    if (streamViewFlag) {
        document.querySelectorAll('.FlipStreamPreviewBox').forEach(async x => {
            if (x.src != "") {
                x.src = `/flipstreamviewer/preview?label=${x.name}&mtime=${data.preview_mtime[x.id] || 0}`;
            }
        });

        const response = await fetch('/flipstreamviewer/stream/info');
        streamInfo = await response.json();
        if (streamInfo.mtime != streamMTime) {
            streamMTime = streamInfo.mtime;
            clearInterval(updateStreamInterval);
            streamIndex = 0;
            streamCache = await Promise.all(
                Array.from({ length: streamInfo.count }, (_, j) => 
                    new Promise(resolve => {
                        const img = new Image();
                        img.onload = () => resolve(img);
                        img.onerror = () => resolve(img);
                        img.src = `/flipstreamviewer/stream/${j}.png?mtime=${streamInfo.mtime}`;
                    })
                )
            );
            updateStreamInterval = setInterval(updateStreamView, 1000 / streamInfo.fps);
        }
    }
    if (data.update_and_reload) {
        updateParam(true);
    }
}
setInterval(refreshView, 1000);
"""

SCRIPT_CAPTURE=r"""
let capturedCanvas = document.createElement("canvas");
async function capture() {
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
}

function updateCanvas() {
    document.getElementById("captureDialog").style.display = "flex";
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const _capture_offsetX = parseInt(document.getElementById("offsetXRange").value, 10);
    const _capture_offsetY = parseInt(document.getElementById("offsetYRange").value, 10);
    const _capture_scale = parseInt(document.getElementById("scaleRange").value, 10) / 100;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(_capture_scale, _capture_scale);
    ctx.translate(_capture_offsetX, _capture_offsetY);
    ctx.drawImage(capturedCanvas, 0, 0);
    ctx.restore();
}

function closeCaptureDialog() {
    document.getElementById("captureDialog").style.display = "none";
}

function resetPos() {
    document.getElementById("offsetXRange").value = 0;
    document.getElementById("offsetYRange").value = 0;
    document.getElementById("scaleRange").value = 100;
    updateCanvas();
}

async function setFrame() {
    const canvas = document.getElementById("canvas");
    await fetch("/flipstreamviewer/set_frame", {
        method: "POST",
        body: canvas.toDataURL("image/png"),
        headers: { "Content-Type": "image/png" }
    });
    closeCaptureDialog();
}
"""

SCRIPT_QUERY=r"""
function onInputDarker(value=-1) {
    const container = document.getElementById('mainDialog');
    value = parseFloat(value);
    if (value < 0) {
        value = document.getElementById("darkerRange").value;
    } else {
        document.getElementById("darkerRange").value = value;
    }
    document.getElementById("darkerValue").innerText = value; 
    container.style.backgroundColor = 'rgba(0,0,0,' + value + ')';
}
onInputDarker();

function reloadPage(search="") {
    if (search) {
        location.replace(location.pathname + "?" + search);
    } else {
        location.reload();
    }
}

function parseQueryParam() {
    const p = new URLSearchParams(location.search)
    if (p.has("px")) {
        document.getElementById("presetXorkeyInputDiv").style.display = "block";
    }
    if (p.has("darker")) {
        onInputDarker(p.get("darker"));
    }
    if (p.has("toggleView")) {
        toggleView();
    }
    if (p.has("showPresetDialog")) {
        showPresetDialog();
    }
    if (p.has("presetFolder")) {
        const presetFolder = document.getElementById("presetFolderSelect").value;
        if (presetFolder != p.get("presetFolder")) {
            updateParam(true, { presetFolder: p.get("presetFolder") }, {}, p.toString());
            return;
        }
    }
    if (p.has("presetFile")) {
        const presetFile = document.getElementById("presetFileSelect").value;
        if (presetFile != p.get("presetFile")) {
            loadPreset(false, { presetFile: p.get("presetFile") }, p.toString());
            return;
        }
    }
}
parseQueryParam();
"""


@server.PromptServer.instance.routes.get("/flipstreamviewer/stream/info")
async def stream_count(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    return web.json_response({"mtime": frame_mtime, "fps": frame_fps, "count": len(frame_buffer)})


@server.PromptServer.instance.routes.get("/flipstreamviewer/stream/{frame_id:\\d+}.png")
async def stream_frame(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    try:
        frame_id = int(request.match_info["frame_id"])
    except (KeyError, ValueError):
        raise web.HTTPBadRequest(text="Invalid frame ID")

    if not (0 <= frame_id < len(frame_buffer)):
        raise web.HTTPNotFound(text="Frame not found")
    
    return web.Response(body=frame_buffer[frame_id], headers={"Content-Type": "image/png"})


@server.PromptServer.instance.routes.get("/flipstreamviewer/preview")
async def preview(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    label = request.rel_url.query.get('label', '')
    key = label + "PreviewBox"
    if key in state:
        data = state[key][1]
    else:
        data = b""
    return web.Response(body=data, headers={"Content-Type": "image/png"})


@server.PromptServer.instance.routes.post("/flipstreamviewer/preview_setroi")
async def preview_setroi(request):
    if request.remote not in allowed_ips: raise HTTPForbidden()
    data = await request.json()
    param[data['label'] + "PreviewRoi"] = data
    return web.Response()


@server.PromptServer.instance.routes.get("/flipstreamviewer")
async def viewer(request):
    if request.remote not in allowed_ips:
        print(request.remote)
        raise HTTPForbidden()

    block = {}

    def add_section(title, section, hook):
        block[f"{title}_{section}"] = f"""
          </details>
          <details>
            <summary><i>{section}</i></summary>"""

    def add_button(title, capture, update, hook):
        block[f"{title}"] = f"""
            <div class="row">"""
        if capture:
            block[f"{title}"] += f"""
                <button onclick="capture()">Capture</button>"""
        if update:
            block[f"{title}"] += f"""
                <button class="willreload" onclick="updateParam(true)">Update</button>"""
        block[f"{title}"] += f"""
            </div>"""

    def add_slider(title, label, default, min, max, step):
        if not label.isidentifier():
            raise RuntimeError(f"{title}: label must contain only valid identifier characters.")
        param.setdefault(label, default)
        block[f"{title}_{label}"] = f"""
            <div class="row" style="color: lightslategray;">
                <input class="FlipStreamSlider" id="{label}Slider" type="range" min="{min}" max="{max}" step="{step}" name="{label}" value="{float(param[label]):g}" oninput="{label}Value.innerText = this.value;" />
                <span id="{label}Value">{float(param[label]):g}</span>{label}
            </div>"""

    def add_textbox(title, label, default, rows):
        if not label.isidentifier():
            raise RuntimeError(f"{title}: label must contain only valid identifier characters.")
        param.setdefault(label, btoa_utf8(default))
        block[f"{title}_{label}"] = f"""
            <textarea class="FlipStreamTextBox" id="{label}TextBox" style="color: lightslategray;" placeholder="{label}" rows="{rows}" name="{label}">{atob_utf8(param[label])}</textarea>"""

    def add_inputbox(title, label, default, boxtype):
        if not label.isidentifier():
            raise RuntimeError(f"{title}: label must contain only valid identifier characters.")
        param.setdefault(label, default)
        if boxtype == "seed":
            block[f"{title}_{label}"] = f"""
            <div class="row" style="color: lightslategray;">
                {label}: <input class="FlipStreamInputBox" id="{label}InputBox" style="color: lightslategray;" placeholder="{label}" type="number" name="{label}" value="{param[label]}" />
                <button onclick="{label}InputBox.value=Math.floor(Math.random()*1e7); updateParam(true)">R</button>
            </div>"""
        elif boxtype == "r4d":
            block[f"{title}_{label}"] = f"""
            <div class="row" style="color: lightslategray;">
                {label}: <input class="FlipStreamInputBox" id="{label}InputBox" style="color: lightslategray;" placeholder="{label}" type="number" name="{label}" value="{param[label]}" />
                <button onclick="{label}InputBox.value=Math.floor(Math.random()*1e4); updateParam(true)">R</button>
            </div>"""
        else:
            block[f"{title}_{label}"] = f"""
            <div class="row" style="color: lightslategray;">
                {label}: <input class="FlipStreamInputBox" id="{label}InputBox" style="color: lightslategray;" placeholder="{label}" type="{boxtype}" name="{label}" value="{param[label]}" />
                <button onclick="updateParam(true)">U</button>
            </div>"""

    def add_selectbox(title, label, default, listitems):
        listitems = listitems.split(",")
        if not label.isidentifier():
            raise RuntimeError(f"{title}: label must contain only valid identifier characters.")
        if not all(item == html.escape(item) for item in listitems):
            raise RuntimeError(f"{title}: listitems must contain only HTML-acceptable characters.")
        param.setdefault(label, "")
        text_html = f"""
            <select class="FlipStreamSelectBox" id="{label}SelectBox" name="{label}">
                <option value="" disabled selected>{label}</option>"""
        for item in listitems:
            text_html += f"""
                <option value="{item}"{" selected" if param[label] == item else ""}>{item}</option>"""
        text_html += f"""
            </select>"""
        block[f"{title}_{label}"] = text_html

    def add_fileselect(title, label, default, folder_name, folder_path, mode, use_sub, use_move):
        if not label.isidentifier():
            raise RuntimeError(f"{title}: label must contain only valid identifier characters.")
        if not (mode == "" or mode.isidentifier()):
            raise RuntimeError(f"{title}: mode must contain only valid identifier characters.")
        param.setdefault(label, "")
        state.setdefault(f"{label}Folder", "")
        text_html = ""
        if use_sub:
            text_html += f"""
            <select class="FlipStreamFolderSelect willreload" id="{label}FolderSelect" name="{label}Folder" onchange="updateParam(true)">
                <option value="">{label} folder</option>"""
            for dir in sorted(Path(folder_path, mode).glob("*/")):
                selected = " selected" if state[f"{label}Folder"] == dir.name else ""
                text_html += f"""
                <option value="{dir.name}"{selected}>{dir.name}</option>"""
            text_html += f"""
            </select>"""
        text_html += f"""
            <div class="row">
                <select class="FlipStreamFileSelect" id="{label}FileSelect" name="{label}">
                    <option value="">{label} default</option>"""
        if param[label]:
            text_html += f"""
                    <option value="{param[label]}" selected>*{Path(param[label]).stem}</option>"""
        if use_sub:
            files = sorted(Path(folder_path, mode, state[f"{label}Folder"]).glob("*.*"))
            files = [Path(file).relative_to(folder_path) for file in files]
        else:
            files = [Path(file) for file in FlipStreamFileSelect.get_filelist(folder_name, folder_path, mode)]
        
        png_files = [file for file in files if file.suffix.lower() == '.png']
        other_files = [file for file in files if file.suffix.lower() != '.png']
        
        for file in other_files:
            if file != param[label]:
                text_html += f"""
                    <option value="{file}">{file.stem}</option>"""
                png_file = file.with_suffix('.png')
                if png_file in png_files:
                    png_files.remove(png_file)
        for file in png_files:
            if file != param[label]:
                text_html += f"""
                    <option value="{file}">{file.stem}</option>"""
        text_html += f"""
                </select>"""
        if use_move:
            text_html += f"""
                <button class="willreload" id="{label}MoveButton" onClick="showMoveFileSelect({label}MoveFileSelect)">M</button>"""
        text_html += f"""
            </div>"""
        if use_move:
            text_html += f"""
            <select class="FlipStreamMoveFileSelect" id="{label}MoveFileSelect" onchange="moveFile(String.raw`{folder_path}`, '{mode}', '{label}')">
                <option value="" disabled selected>move to</option>"""
            for dir in Path(folder_path, mode).glob("*/"):
                text_html += f"""
                <option value="{dir.name}">{dir.name}</option>"""
            text_html += f"""
            </select>"""
        block[f"{title}_{label}"] = text_html

    def add_previewbox(title, label, tensor):
        if not label.isidentifier():
            raise RuntimeError(f"{title}: label must contain only valid identifier characters.")
        if (label + "PreviewBox") in state:
            block[f"{title}_{label}"] = f"""
            <div class="row" style="color: lightslategray;">
                <img class="FlipStreamPreviewBox" id="{label}PreviewBox" name="{label}" src="/flipstreamviewer/preview?label={label}" alt onerror="this.onerror = null; this.src='';" />
                <canvas class="FlipStreamPreviewRoi" id="{label}PreviewRoi"></canvas>
            </div>"""
    
    def add_logbox(title, label, log, rows):
        if not label.isidentifier():
            raise RuntimeError(f"{title}: label must contain only valid identifier characters.")
        if (label + "LogBox") in state:
            block[f"{title}_{label}"] = f"""
            <textarea class="FlipStreamLogBox" id="{label}LogBox" style="color: lightslategray;" placeholder="{label}" rows="{rows}" name="{label}" readonly></textarea>"""

    hist = server.PromptServer.instance.prompt_queue.get_history(max_items=1)
    nodedict = next(iter(hist.values()))["prompt"][2] if hist else None
    if nodedict:
        for node in nodedict.values():
            class_type = node["class_type"]
            title = node["_meta"]["title"]
            inputs = node["inputs"]
            if class_type == "FlipStreamSection":
                add_section(title, **inputs)
            if class_type == "FlipStreamButton":
                add_button(title, **inputs)
            if class_type == "FlipStreamSlider":
                add_slider(title, **inputs)
            if class_type == "FlipStreamTextBox":
                add_textbox(title, **inputs)
            if class_type == "FlipStreamInputBox":
                add_inputbox(title, **inputs)
            if class_type.startswith("FlipStreamSelectBox"):
                add_selectbox(title, **inputs)
            if class_type.startswith("FlipStreamFileSelect"):
                add_fileselect(title, **inputs)
            if class_type == "FlipStreamPreviewBox":
                add_previewbox(title, **inputs)
            if class_type == "FlipStreamLogBox":
                add_logbox(title, **inputs)

    text_html = f"""<html>{HEAD}<body>
    <div id="mainDialog">
        <div id="leftPanel">
          <details>
            <summary><i>Input</i></summary>
            {"".join([x[1] for x in sorted(block.items())])}
          </details>
        </div>
        <div id="centerPanel" onclick="toggleView()">
            <div id="messageBox"></div>
        </div>
        <div id="rightPanel">
          <details>
            <summary><i>Status</i></summary>
            <textarea id="statusInfo" style="color: lightslategray;" rows="5"></textarea>
            <div class="row">
                <input id="darkerRange" type="range" min="0" max="1" step="0.01" value="{state["darker"]}" oninput="onInputDarker();" />
                <span id="darkerValue">{state["darker"]}</span>drk
            </div>
          </details>
          <details>
            <summary><i>Preset</i></summary>
            <select id="presetFolderSelect" class="willreload" onchange="updateParam(true)">
                <option value="" selected>preset folder</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["presetFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("preset").glob("*/")])}
            </select>
            <div class="row">
                <select id="presetFileSelect">
                    <option value="" disabled selected>preset</option>
                    {"".join([f'<option value="{file.name}"{" selected" if state["presetFile"] == file.name else ""}>{file.stem}</option>' for file in Path("preset", state["presetFolder"]).glob("*.json")])}
                </select>
                <button onclick="movePreset()">M</button>
            </div>
            <select id="movePresetSelect" onchange="movePresetFile()">
                <option value="" disabled selected>move to</option>
                {"".join([f'<option value="{dir.name}">{dir.name}</option>' for dir in Path("preset").glob("*/")])}
            </select>
            <div class="row">
                <input id="presetTitleInput" placeholder="preset title" value="{state["presetTitle"]}" />
                <button onclick="savePreset()">Save</button>
            </div>
            <div id="presetXorkeyInputDiv">
                <input id="presetXorkeyInput" type="password" placeholder="xorkey" />
            </div>
            <div class="row">
                <button onclick="showPresetDialog()">Choose</button>
                <button class="willreload" onclick="loadPreset()">Load</button>
                <button class="willreload" onclick="loadPreset(true)">LoraOnly</button>
            </div>
          </details>
          <details>
            <summary><i>Lora</i></summary>
            <select id="loraFolderSelect" class="willreload" onchange="updateParam(true)">
                <option value="" selected>lora folder</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["loraFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("ComfyUI/models/loras", state["loraMode"]).glob("*/")])}
            </select>
            <div class="row">
                <select id="loraFileSelect" onchange="selectLoraFile()">
                    <option value="" disabled selected>lora file</option>
                    {"".join([f'<option value="{file.name}"{" selected" if state["loraFile"] == file.name else ""}>{file.stem}</option>' for file in Path("ComfyUI/models/loras", state["loraMode"], state["loraFolder"]).glob("*.safetensors")])}
                </select>
                <button onclick="toggleLora()">T</button>
                <button onClick="moveLora()">M</button>
            </div>
            <select id="moveLoraSelect" onchange="moveLoraFile()">
                <option value="" disabled selected>move to</option>
                {"".join([f'<option value="{dir.name}">{dir.name}</option>' for dir in Path("ComfyUI/models/loras", state["loraMode"]).glob("*/")])}
            </select>
            <div class="row">
                <select id="loraTagSelect" onchange="toggleTag()">
                    <option value="">tags</option>
                    {"".join([f'<option value="{opt["value"]}"{" selected" if state["loraTag"] == opt["value"] else ""}>{opt["text"]}</option>' if opt["value"] else "" for opt in json.loads(state["loraTagOptions"])])}
                </select>
                <button onclick="toggleTag()">T</button>
                <button onclick="randomTag()">R</button>
            </div>
            Rate: <select id="loraRate">
                <option value="8" {"selected" if state["loraRate"] == "8" else ""}>8</option>
                <option value="7" {"selected" if state["loraRate"] == "7" else ""}>7</option>
                <option value="6" {"selected" if state["loraRate"] == "6" else ""}>6</option>
                <option value="5" {"selected" if state["loraRate"] == "5" else ""}>5</option>
                <option value="4" {"selected" if state["loraRate"] == "4" else ""}>4</option>
                <option value="3" {"selected" if state["loraRate"] == "3" else ""}>3</option>
                <option value="2" {"selected" if state["loraRate"] == "2" else ""}>2</option>
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
                <option value="0" {"selected" if state["loraRate"] == "0" else ""}>0</option>
                <option value="-1" {"selected" if state["loraRate"] == "-1" else ""}>-1</option>
                <option value="-2" {"selected" if state["loraRate"] == "-2" else ""}>-2</option>
                <option value="-3" {"selected" if state["loraRate"] == "-3" else ""}>-3</option>
                <option value="-4" {"selected" if state["loraRate"] == "-4" else ""}>-4</option>
                <option value="-5" {"selected" if state["loraRate"] == "-5" else ""}>-5</option>
            </select>
            <select id="loraRank">
                <option value="0" {"selected" if state["loraRank"] == "0" else ""}>0</option>
                <option value="10" {"selected" if state["loraRank"] == "10" else ""}>10</option>
                <option value="20" {"selected" if state["loraRank"] == "20" else ""}>20</option>
                <option value="30" {"selected" if state["loraRank"] == "30" else ""}>30</option>
                <option value="40" {"selected" if state["loraRank"] == "40" else ""}>40</option>
                <option value="50" {"selected" if state["loraRank"] == "50" else ""}>50</option>
            </select>
            <div class="row">
                <button onclick="showTagDialog()">Choose</button>
                <button onclick="updateParam(true)">Update</button>
                <button onclick="clearLoraInput()">Clr</button>
                <button onclick="showTagDialog();tagCSel();tagRSel();tagOK();updateParam(true)">R</button>
            </div>
            <textarea id="loraInput" placeholder="Enter lora" rows="12">{atob_utf8(param["lora"])}</textarea>
            <div class="row">
                <a id="loraLink" href="{state["loraLinkHref"] or "javascript:void(0)"}" target="_blank">
                    <img id="loraPreview" src="{state["loraPreviewSrc"]}" alt onerror="this.onerror = null; this.src='';" />
                </a>
            </div>
          </details>
        </div>
    </div>
    <div id="captureDialog">
        <div id="captureLeftPanel">
        </div>
        <div id="captureCenterPanel">
            <canvas id="canvas" width="960" height="600" />
        </div>
        <div id="captureRightPanel">
            <div class="row">
                <button onclick="resetPos()">Reset</button>
            </div>
            <div class="row">
                X <input id="offsetXRange" type="range" min="-960" max="960" value="{param["_capture_offsetX"]}" onchange="updateCanvas()">
            </div>
            <div class="row">
                Y <input id="offsetYRange" type="range" min="-600" max="600" value="{param["_capture_offsetY"]}" onchange="updateCanvas()">
            </div>
            <div class="row">
                S <input id="scaleRange" type="range" min="50" max="400" value="{param["_capture_scale"]}" onchange="updateCanvas()">
            </div>
            <div class="row">
                <button onclick="setFrame()">SetFrame</button>
                <button onclick="closeCaptureDialog()">Close</button>
            </div>
        </div>
    </div>
    <div id="tagDialog">
        <div id="tagLeftPanel">
        </div>
        <div id="tagCenterPanel">
        </div>
        <div id="tagRightPanel">
            <div class="row">
                <button onclick="tagCSel()">CSel</button>
                <button onclick="tagRSel()">RSel</button>
            </div>
            <div class="row">
                <button onclick="tagOK()">OK</button>
                <button onclick="closeTagDialog()">Cancel</button>
            </div>
        </div>
    </div>
    <div id="presetDialog">
        <div id="presetLeftPanel">
        </div>
        <div id="presetCenterPanel" onclick="toggleView()">
            <div id="messageBox"></div>
        </div>
        <div id="presetRightPanel">
            <div class="row">
                <button onclick="closePresetDialog()">Close</button>
            </div>
            <div class="row"><i>Status</i></div>
            <div id="statusInfo"></div>
            <div class="row"><i>Darker</i></div>
            <div class="row">
                <input id="darkerRange" type="range" min="0" max="1" step="0.01" value="{state["darker"]}" oninput="onInputDarker(this.value);" />
                <span id="darkerValue">{state["darker"]}</span>drk
            </div>
            <div class="row"><i>Preset</i></div>
            {"".join([f'''
            <button onclick="loadPreset(false, {{ presetFile: '{file.name}' }}, 'showPresetDialog')">{file.stem}</button>
            ''' for file in Path("preset", state["presetFolder"]).glob("*.json")])}
        </div>
    </div>
    <script>
    {SCRIPT_PARAM}
    {SCRIPT_PRESET}
    {SCRIPT_CKPT}
    {SCRIPT_LORA}
    {SCRIPT_TAG}
    {SCRIPT_VIEW}
    {SCRIPT_CAPTURE}
    {SCRIPT_QUERY}
    </script>
    </body></html>"""
    return web.Response(text=text_html, content_type="text/html")


@server.PromptServer.instance.routes.get("/flipstreamviewer/refresh_view")
async def get_status(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    # Get status info from history
    remain = server.PromptServer.instance.prompt_queue.get_tasks_remaining()
    hist = server.PromptServer.instance.prompt_queue.get_history(max_items=1)
    status_info = []
    if frame_updating:
        status_info.append(f"updating {int(time.time() - frame_updating)}s")
    status_info.append(f"q{remain}")
    info = next(iter(hist.values()))["status"] if hist else None
    if info:
        errinfo = info["messages"][2][1]
        status_info.append(info["status_str"])
        status_info += [errinfo[key] for key in ["node_id", "node_type", "exception_message", "exception_type"] if key in errinfo]

    data = refresh_data.copy()
    data["param"] = refresh_param.copy()
    param.update(refresh_param)
    refresh_param.clear()
    data["status_info"] = status_info
    data["preview_mtime"] = {key: state[key][0] for key in state if key.endswith("PreviewBox")}
    data["log"] = {key: state[key] for key in state if key.endswith("LogBox")}
    return web.json_response(data)


@server.PromptServer.instance.routes.post("/flipstreamviewer/update_param")
async def update_param(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, prm = await request.json()
    state.update(stt)
    param.update(prm)
    time.sleep(UPDATE_DELAY)
    return web.Response()


@server.PromptServer.instance.routes.post("/flipstreamviewer/get_lorainfo")
async def get_lorainfo(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    req = await request.json()
    loraPath = Path("ComfyUI/models/loras", state["loraMode"], req["loraFolder"], req["loraFile"])
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

    if hash:
        res = requests.get("https://civitai.com/api/v1/model-versions/by-hash/" + hash).json()
        lorainfo["_lorapreview_href"] = "https://civitai.com/models/" + str(res["modelId"])
        lorainfo["_lorapreview_src"] = res["images"][0]["url"]

    return web.json_response(lorainfo)


@server.PromptServer.instance.routes.post("/flipstreamviewer/set_frame")
async def set_frame(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    frame = base64.b64decode((await request.text()).split(',', 1)[1])
    global frame_buffer
    global frame_mtime
    global setframe_mtime
    global setframe_buffer
    frame_buffer = [frame]
    frame_mtime = time.time()
    setframe_buffer = [frame]
    setframe_mtime = frame_mtime
    return web.Response()


@server.PromptServer.instance.routes.post("/flipstreamviewer/move_file")
async def move_file(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, folder_path, mode, label, moveFrom, moveTo = await request.json()
    state.update(stt)
    pathFrom = Path(folder_path, mode, state[label + "Folder"], Path(moveFrom).name)
    pathTo = Path(folder_path, mode, moveTo, pathFrom.name)
    Path(pathFrom).rename(pathTo)
    state[label + "Folder"] = moveTo
    return web.Response()


@server.PromptServer.instance.routes.post("/flipstreamviewer/move_presetfile")
async def move_presetfile(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, moveTo = await request.json()
    state.update(stt)
    pathFrom = Path("preset", state["presetFolder"], state["presetFile"])
    pathTo = Path("preset", moveTo, pathFrom.name)
    Path(pathFrom).rename(pathTo)
    state["presetFolder"] = moveTo
    return web.Response()


@server.PromptServer.instance.routes.post("/flipstreamviewer/move_lorafile")
async def move_lorafile(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, moveTo = await request.json()
    state.update(stt)
    pathFrom = Path("ComfyUI/models/loras", state["loraMode"], state["loraFolder"], state["loraFile"])
    pathTo = Path("ComfyUI/models/loras", state["loraMode"], moveTo, pathFrom.name)
    Path(pathFrom).rename(pathTo)
    state["loraFolder"] = moveTo
    return web.Response()


def xor_crypt(data_str, key_str, crypt=True):
    xorall = lambda data, key: bytes([b1 ^ b2 for b1, b2 in zip(data, itertools.cycle(key))])
    key_bytes = key_str.encode('utf-8')
    input_bytes = data_str.encode('utf-8')
    if crypt:
        encrypted_bytes = xorall(input_bytes, key_bytes)
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    else:
        decoded_bytes = base64.b64decode(input_bytes)
        return xorall(decoded_bytes, key_bytes).decode('utf-8')


@server.PromptServer.instance.routes.post("/flipstreamviewer/load_preset")
async def load_preset(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, loraPromptOnly, xorkey = await request.json()
    filename = stt["presetFile"]
    path = Path("preset", stt["presetFolder"], filename)
    with open(path, "r") as file:
        if xorkey:
            buf = json.loads(xor_crypt(file.read(), xorkey + filename, False))
        else:
            buf = json.load(file)
        if loraPromptOnly:
            buf = {"lora": buf.get("lora", "")}

    state.update(stt)
    param.clear()
    param.update(default_param)
    param.update(buf)
    state["presetTitle"] = Path(state["presetFile"]).stem
    time.sleep(UPDATE_DELAY)
    return web.json_response({"lora": param["lora"], "presetTitle": state["presetTitle"]})


@server.PromptServer.instance.routes.post("/flipstreamviewer/save_preset")
async def save_preset(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, prm, xorkey = await request.json()
    state.update(stt)
    param.update(prm)
    time.sleep(UPDATE_DELAY)

    filename = state["presetTitle"] + ".json"
    path = Path("preset", state["presetFolder"], filename)
    with open(path, "w") as file:
        if xorkey:
            file.write(xor_crypt(json.dumps(param), xorkey + filename))
        else:
            json.dump(param, file)
    return web.Response()


class FlipStreamSection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "section": ("STRING", {"default": "Section"}),
            },
            "optional": {
                "hook": (any,),
            }
        }

    RETURN_TYPES = (any,)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, section, hook=None):
        return (hook,)


class FlipStreamButton:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "capture": ("BOOLEAN", {"default": True}),
                "update": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "hook": (any,),
            }
        }

    RETURN_TYPES = (any,)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, hook=None, **kwargs):
        return (hook,)


class FlipStreamSlider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "default": ("FLOAT", {"default": 0, "step": 0.01, "round": 0.01}),
                "min": ("FLOAT", {"default": 0, "step": 0.01, "round": 0.01}),
                "max": ("FLOAT", {"default": 0, "step": 0.01, "round": 0.01}),
                "step": ("FLOAT", {"default": 0.01, "min": 0.00, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "BOOLEAN")
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, label, default, **kwargs):
        param.setdefault(label, default)
        return hash((param[label],))
    
    def run(self, label, default, **kwargs):
        global frame_updating
        frame_updating = time.time()
        param.setdefault(label, default)
        return (float(param[label]), int(float(param[label])), bool(float(param[label])))


class FlipStreamTextBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "default": ("STRING", {"default": "", "multiline": True}),
                "rows": ("INT", {"default": 3}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, label, default, **kwargs):
        param.setdefault(label, btoa_utf8(default))
        return hash((param[label],))
    
    def run(self, label, default, **kwargs):
        global frame_updating
        frame_updating = time.time()
        param.setdefault(label, btoa_utf8(default))
        return (atob_utf8(param[label]),)


class FlipStreamInputBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "default": ("STRING", {"default": ""}),
                "boxtype": (["text", "number", "seed", "r4d"],),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN")
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, label, default, **kwargs):
        param.setdefault(label, default)
        return hash((param[label],))

    def floator0(self, v):
        try:
            return float(v)
        except:
            return 0
    
    def run(self, label, default, boxtype, **kwargs):
        global frame_updating
        frame_updating = time.time()
        param.setdefault(label, default)
        t = param[label]
        v = self.floator0(t)
        return (t, v, int(v), bool(t if boxtype == "text" else v))


class FlipStreamSelectBox:
    LISTITEMS = ["a", "b", "c"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "default": (s.LISTITEMS,),
                "listitems": ("STRING", {"default": ",".join(s.LISTITEMS)})
            },
        }

    RETURN_TYPES = (any, "BOOLEAN",)
    RETURN_NAMES = ("item", "enable",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, label, **kwargs):
        param.setdefault(label, "")
        return hash((param[label],))

    def run(self, label, default, **kwargs):
        global frame_updating
        frame_updating = time.time()
        param.setdefault(label, "")
        item = param[label] if param[label] else default
        return (item, param[label] != "")


class FlipStreamSelectBox_Samplers(FlipStreamSelectBox):
    LISTITEMS = comfy.samplers.KSampler.SAMPLERS


class FlipStreamSelectBox_Scheduler(FlipStreamSelectBox):
    LISTITEMS = comfy.samplers.KSampler.SCHEDULERS


class FlipStreamFileSelect:
    FOLDER_NAME = ""
    FOLDER_PATH = ""

    @staticmethod
    def get_filelist(folder_name, folder_path, mode):
        if folder_name == "checkpoints":
            return CheckpointLoaderSimple.INPUT_TYPES()["required"]["ckpt_name"][0]
        elif folder_name == "vae":
            return VAELoader.INPUT_TYPES()["required"]["vae_name"][0]
        elif folder_name == "controlnet":
            return folder_paths.get_filename_list("controlnet")
        elif folder_name == "tensorrt":
            if folder_path.startswith("_error_"):
                return [folder_path]
            if TensorRTLoader is None:
                return ["_error_ ComfyUI_TensorRT is not installed"]
            return TensorRTLoader.INPUT_TYPES()["required"]["unet_name"][0]
        elif folder_name == "animatediff_models":
            if folder_path.startswith("_error_"):
                return [folder_path]
            return folder_paths.get_filename_list(folder_name)
        else:
            return [str(p.relative_to(folder_path)) for p in Path(folder_path, mode).glob("*.*")]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "default": ([""] + s.get_filelist(s.FOLDER_NAME, s.FOLDER_PATH, ""),),
                "folder_name": ([s.FOLDER_NAME],),
                "folder_path": ([s.FOLDER_PATH],),
                "mode": ("STRING", {"default": ""}),
                "use_sub": ("BOOLEAN", {"defalut": False}),
                "use_move": ("BOOLEAN", {"defalut": False}),
            }
        }

    RETURN_TYPES = (any, any, "BOOLEAN",)
    RETURN_NAMES = ("file", "path", "enable",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, label, **kwargs):
        param.setdefault(label, "")
        return hash((param[label],))

    def run(self, label, default, folder_path, **kwargs):
        global frame_updating
        frame_updating = time.time()
        param.setdefault(label, "")
        file = param[label] if param[label] else default
        return (file, str(Path(folder_path, file)), param[label] != "")


class FlipStreamFileSelect_Checkpoints(FlipStreamFileSelect):
    FOLDER_NAME = "checkpoints"
    FOLDER_PATH = Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()).as_posix()


class FlipStreamFileSelect_Loras(FlipStreamFileSelect):
    FOLDER_NAME = "loras"
    FOLDER_PATH = Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()).as_posix()


class FlipStreamFileSelect_VAE(FlipStreamFileSelect):
    FOLDER_NAME = "vae"
    FOLDER_PATH = Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()).as_posix()


class FlipStreamFileSelect_ControlNetModel(FlipStreamFileSelect):
    FOLDER_NAME = "controlnet"
    FOLDER_PATH = Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()).as_posix()


class FlipStreamFileSelect_TensorRT(FlipStreamFileSelect):
    FOLDER_NAME = "tensorrt"
    try:
        FOLDER_PATH = Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()).as_posix()
    except:
        FOLDER_PATH = "_error_ tensorrt folder is not found"


class FlipStreamFileSelect_AnimateDiffModel(FlipStreamFileSelect):
    FOLDER_NAME = "animatediff_models"
    try:
        FOLDER_PATH = Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()).as_posix()
    except:
        FOLDER_PATH = "_error_ animatediff_models folder is not found"


class FlipStreamFileSelect_Input(FlipStreamFileSelect):
    FOLDER_NAME = "input"
    FOLDER_PATH = Path(folder_paths.input_directory).relative_to(Path.cwd()).as_posix()


class FlipStreamFileSelect_Output(FlipStreamFileSelect):
    FOLDER_NAME = "output"
    FOLDER_PATH = Path(folder_paths.output_directory).relative_to(Path.cwd()).as_posix()


class FlipStreamPreviewBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "tensor": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, label, tensor, **kwargs):
        if tensor is None:
            tensor = torch.zeros((1, 8, 8, 3))
        buf = np.array(tensor[0].cpu().numpy() * 255, dtype=np.uint8)
        image = Image.fromarray(buf)
        image.thumbnail((256, 256))
        with io.BytesIO() as output:
            image.save(output, format="PNG", compress_level=STREAM_COMPRESSION)
            state[label + "PreviewBox"] = (time.time(), output.getvalue())
        return ()


class FlipStreamLogBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "log": ("STRING",),
                "rows": ("INT", {"default": 3}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, label, log, **kwargs):
        state[label + "LogBox"] = btoa_utf8(log)
        return ()


class FlipStreamSetUpdateAndReload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "delay_sec": ("FLOAT", {"default": 1, "min": 0.0, "max": 30.0, "step": 0.1}),
            },
            "optional": {
                "hook": (any,),
            }
        }

    RETURN_TYPES = (any,)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, delay_sec, hook=None):
        time.sleep(delay_sec)
        refresh_data["update_and_reload"] = True
        return (hook,)


class FlipStreamSetMessage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "message": ("STRING", {"default": "", "multiline": True}),
                "fontsize": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "hook": (any,),
            }
        }

    RETURN_TYPES = (any,)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, message, fontsize, hook=None):
        refresh_data["message"] = btoa_utf8(message)
        refresh_data["message_fontsize"] = f"{fontsize}rem"
        return (hook,)


class FlipStreamSetParam:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "value": ("STRING", {"default": "", "multiline": True}),
                "replace": ("BOOLEAN", {"default": False}),
                "b64enc": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "hook": (any,),
            }
        }

    RETURN_TYPES = (any,)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, label, value, replace, b64enc, hook=None):
        empty = ""
        if b64enc:
            value = btoa_utf8(value)
            empty = btoa_utf8(empty)
        if replace or label not in param or param[label] == empty:
            refresh_param[label] = value
        return (hook,)


class FlipStreamGetParam:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "default": ("STRING", {"default": ""}),
                "b64dec": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, label, default, b64dec):
        value = default
        if label in param:
            value = param[label]
            if b64dec:
                value = atob_utf8(value)
        return hash((value,))

    def run(self, label, default, b64dec):
        global frame_updating
        frame_updating = time.time()
        value = default
        if label in param:
            value = param[label]
            if b64dec:
                value = atob_utf8(value)
        return (value,)


class FlipStreamGetPreviewRoi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32}),
                "default_left": ("INT", {"default": 0}),
                "default_top": ("INT", {"default": 0}),
                "default_right": ("INT", {"default": 0}),
                "default_bottom": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("left", "top", "right", "bottom", "inner_width", "inner_height")
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, label, **kwargs):
        roi_data = frozenset(param.get(label + "PreviewRoi", {}).items())
        return hash(roi_data)

    def run(self, label, default_left, default_top, default_right, default_bottom, width, height):
        global frame_updating
        frame_updating = time.time()
        roi_data = param.get(label + "PreviewRoi", {})
        left = int(roi_data['sx'] * width) if 'sx' in roi_data else default_left
        top = int(roi_data['sy'] * height) if 'sy' in roi_data else default_top
        right = int(width - roi_data['ex'] * width) if 'ex' in roi_data else default_right
        bottom = int(height - roi_data['ey'] * height) if 'ey' in roi_data else default_bottom
        inner_width = width - left - right
        inner_height = height - top - bottom
        return (left, top, right, bottom, inner_width, inner_height)


class FlipStreamImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE","INT","INT","INT")
    RETURN_NAMES = ("tensor","width","height","batchsize")
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, tensor):
        (batchsize, height, width) = tensor.shape[0:3]
        return (tensor, width, height, batchsize)


class FlipStreamTextReplace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "find": ("STRING", {"default": ""}),
                "replace": ("STRING", {"default": ""}),
            },
            "optional": {
                "value": (any,),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, text, find, replace, value=None):
        for word in find.split(","):
            word = word.strip()
            if not word:
                continue
            text = text.replace(word, replace.format(value))
        return (text,)


class FlipStreamGetFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0}),
                "frames": ("INT", {"default": 1, "min": 1}),
                "capture_only": ("BOOLEAN", {"default": True}),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "enable")
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, capture_only, **kwargs):
        if capture_only:
            return setframe_mtime
        else:
            return frame_mtime
    
    def run(self, index, frames, capture_only, enable, **kwargs):
        if not enable:
            return (None, False)            
        buf = setframe_buffer if capture_only else frame_buffer
        images_list = [
            np.array(Image.open(io.BytesIO(buf[i]))) 
            for i in range(index, min(index + frames, len(buf)))
        ]        
        if not images_list:
            return (None, False)
        image = torch.from_numpy(np.stack(images_list)).float() / 255.0
        if image is not None and image.shape[3] == 4:
            image = image[:,:,:,:3] * image[:,:,:,3:4]
        return (image, True)


class FlipStreamScreenGrabber:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "left": ("INT", {"default": 0, "min": 0, "step": 32}),
                "top": ("INT", {"default": 0, "min": 0, "step": 32}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32}),
                "frames": ("INT", {"default": 8, "min": 1, "max": 100}),
                "fps": ("INT", {"default": 8, "min": 1, "max": 30}),
                "delay_sec": ("FLOAT", {"default": 3, "min": 0.0, "max": 30.0, "step": 0.1}),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "enable")
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def __init__(self):
        self.grabber_thread = None
        self.flag_stop_grabber = False
        self.last_args = None
        self.grabbed_frames = []

    def grabber(self, area, frames, fps):
        with mss.mss() as sct:
            while True:
                self.grabbed_frames.append(sct.grab(area))  # BGRA format
                self.grabbed_frames = self.grabbed_frames[-frames:]
                time.sleep(1 / fps)
                if self.flag_stop_grabber:
                    break

    def start_grabber(self, area, frames, fps):
        if self.grabber_thread is None or not self.grabber_thread.is_alive():
            self.flag_stop_grabber = False
            self.grabber_thread = threading.Thread(target=self.grabber, args=(area, frames, fps), daemon=True)
            self.grabber_thread.start()

    def stop_grabber(self):
        if self.grabber_thread is not None and self.grabber_thread.is_alive():
            self.flag_stop_grabber = True
            self.grabber_thread.join()

    def __del__(self):
        self.stop_grabber()

    def run(self, top, left, width, height, frames, fps, delay_sec, enable):
        image = None
        if enable:
            enable = False
            area = {"left": left, "top": top, "width": width, "height": height}
            if self.last_args is None or self.last_args != (area, frames, fps):
                self.last_args = (area, frames, fps)
                self.stop_grabber()
            self.start_grabber(area, frames, fps)
            time.sleep(delay_sec)
            buf = self.grabbed_frames[-frames:]
            if len(buf) == frames:
                image = torch.tensor(np.array(buf)[:, :, :, (2, 1, 0)] / 255, dtype=torch.float32)
                enable = True
        else:
            self.stop_grabber()
        return (image, enable)


class FlipStreamVideoInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "first": ("INT", {"default": 0, "min": 0}),
                "step": ("INT", {"default": 1, "min": 1}),
                "frames": ("INT", {"default": 1, "min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "enable")
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, path, first, step, frames):
        if not path or not Path(path).is_file():
            return (torch.zeros((1, 64, 64, 3)), False)
        with iio.imopen(path, "r") as file:
            buf = [file.read(index=i) for i in range(first, first+(frames-1)*step+1, step)]
        buf = np.stack(buf).astype(np.float32) / 255
        enable = buf.shape[0] == frames
        image = torch.from_numpy(buf) if enable else None
        return (image, enable)


class FlipStreamSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32}),
                "frames": ("INT", {"default": 8, "min": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT",)
    RETURN_NAMES = ("image", "latent",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, width, height, frames, image=None, vae=None):
        latent = None
        if image is not None and not torch.any(image):
            image = None            
        if image is not None and image.shape[3] == 4:
            image = image[:,:,:,:3] * image[:,:,:,3:4]
        if image is not None and image.shape[0] >= frames:
            buf = image[:frames]
            buf = buf.movedim(-1,1)
            buf = comfy.utils.common_upscale(buf, width, height, "lanczos", "centor")
            image = buf.movedim(1,-1)
            if vae:
                latent = {"samples": vae.encode(image)}
            else:
                latent = {"samples": torch.zeros([frames, 4, height // 8, width // 8], device=comfy.model_management.intermediate_device())}
        else:
            image = torch.zeros([frames, height, width, 3])
            latent = {"samples": torch.zeros([frames, 4, height // 8, width // 8], device=comfy.model_management.intermediate_device())}
        return (image, latent,)


class FlipStreamSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (any,),
            },
            "optional": {
                "value_enable": (any,),
                "enable": ("BOOLEAN",),
            }
        }

    RETURN_TYPES = (any,)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, value=None, value_enable=None, enable=None):
        if enable:
            return (value_enable,)
        return (value,)


class FlipStreamSwitchImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "image_enable": ("IMAGE",),
                "enable": ("BOOLEAN",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, image=None, image_enable=None, enable=None):
        if enable:
            return (image_enable,)
        return (image,)


class FlipStreamSwitchLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "latent_enable": ("LATENT",),
                "enable": ("BOOLEAN",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, latent, latent_enable=None, enable=None):
        if enable:
            return (latent_enable,)
        return (latent,)


class FlipStreamGate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "latent": ("LATENT",),
            },
            "optional": {
                "a": (any,),
                "b": (any,),
                "c": (any,),
                "d": (any,),
                "e": (any,),
                "f": (any,),
                "g": (any,),
                "h": (any,)
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", any, any, any, any, any, any, any, any)
    RETURN_NAMES = ("model", "pos", "neg", "latent", "a", "b", "c", "d", "e", "f", "g", "h")
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, model, pos, neg, latent, a=None, b=None, c=None, d=None, e=None, f=None, g=None, h=None):
        return (model, pos, neg, latent, a, b, c, d, e, f, g, h)


class FlipStreamRembg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "run"
    CATEGORY = "image"

    def __init__(self):
        if rembg is None:
            raise RuntimeError("FlipStreamRembg: ComfyUI-Inspyrenet-Rembg must be installed to use this function.")

        self.rembg = rembg.InspyrenetRembg()

    def run(self, image):
        img, mask = self.rembg.remove_background(image, "default")
        return (img[..., :3] * img[..., 3:4], mask)


class FlipStreamSegMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("IMAGE",),
                "target": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES =("preview", "mask",) 
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    model = None
    processor = None

    def run(self, tensor, target):
        if florence2 is None:
            raise RuntimeError("FlipStreamSegMask: ComfyUI-Florence2 must be installed to use this function.")

        # Create mask
        _, h, w = tensor.shape[0:3]
        mask_image = Image.new('RGB', (w, h), 'black')

        # Skip if empty target
        if target.strip() == "":
            mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
            return (mask_tensor, mask_tensor[:,:,:,0])

        # Download model if it not found
        model_id = 'microsoft/Florence-2-large'
        if FlipStreamSegMask.model is None:
            model_dir = Path(folder_paths.models_dir, "LLM")
            model_dir.mkdir(exist_ok=True)
            model_name = model_id.rsplit('/', 1)[-1]
            model_path = Path(model_dir, model_name)
            
            if not model_path.exists():
                print(f"Downloading Florence2 model to: {model_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model_id, local_dir=str(model_path), local_dir_use_symlinks=False)
            
        # Load model
        if FlipStreamSegMask.model is None:
            FlipStreamSegMask.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation='sdpa',
                device_map=comfy.model_management.get_torch_device(),
                torch_dtype=torch.bfloat16
            )
            FlipStreamSegMask.processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = FlipStreamSegMask.model
        processor = FlipStreamSegMask.processor
        model.to(comfy.model_management.get_torch_device())

        # Process prompt for each target
        task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
        image = Image.fromarray(np.clip(255. * tensor[0,:,:,:].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        mask_draw = ImageDraw.Draw(mask_image)
        for item in target.split(','):
            prompt = task_prompt + item
            inputs = processor(text=prompt, images=image, return_tensors="pt", do_rescale=False).to('cuda', torch.bfloat16)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"].cuda(),
                pixel_values=inputs["pixel_values"].cuda(),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=1,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(w, h))

            # Draw mask
            predictions = parsed_answer[task_prompt]
            for polygons in predictions['polygons']:
                for p in polygons:
                    p = np.array(p).reshape(-1, 2)
                    p = np.clip(p, [0, 0], [w - 1, h - 1])
                    if len(p) < 3:  
                        print('Invalid polygon:', p)
                        continue

                    p = p.reshape(-1).tolist()
                    mask_draw.polygon(p, outline="white", fill="white")

        # Offload model
        model.to(torch.device("cpu"))

        mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
        return (mask_tensor, mask_tensor[:,:,:,0])


class FlipStreamChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_file": ([path.name for path in Path(folder_paths.models_dir, "LLM").glob("*.gguf")],),
                "n_ctx": ("INT", {"default": 2048, "min": 0, "max": 8192}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1}),
                "unload_other_models": ("BOOLEAN", {"default": False}),
                "close_after_use": ("BOOLEAN", {"default": False}),
                "system": ("STRING", {"default": "", "multiline": True}),
                "user": ("STRING", {"default": "", "multiline": True}),
                "instant": ("BOOLEAN", {"default": False}),
                "max_history": ("INT", {"default": 10, "min": -1}),
                "stop": ("STRING", {"default": "[,<"}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0}),
                "seed": ("INT", {"default": -1}),
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "presence_penalty": ("FLOAT", {"default": 0, "min": 0}),
                "frequency_penalty": ("FLOAT", {"default": 0.5, "min": 0}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0}),
                "response_format": ("STRING", {"default": "", "multiline": True})
            },
            "optional": {
                "chat_model": ("CHAT_MODEL",),
                "messages": ("MESSAGES",)
            }
        }

    RETURN_TYPES = ("CHAT_MODEL", "STRING", "MESSAGES")
    RETURN_NAMES =("chat_model", "response", "messages")
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def __init__(self):
        self.model = None
        self.messages = []
        self.system = None

    def load_model(self, model_file, n_ctx, n_gpu_layers):
        h = hash((model_file, n_ctx, n_gpu_layers))
        if self.model is None or self.model._FlipStreamChat_is_closed or self.model._FlipStreamChat_last_hash != h:
            model_path = Path(folder_paths.models_dir, "LLM", model_file)
            if not model_path.exists():
                raise RuntimeError(f"FlipStreamChat: {model_path} not found.")
            if Llama is None:
                raise RuntimeError("FlipStreamChat: llama-cpp-python required.")
            self.model = Llama(str(model_path), chat_format="llama-2", n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
            self.model._FlipStreamChat_is_closed = False
            self.model._FlipStreamChat_last_hash = h

    def close_model(self):
        self.model.close()
        self.model._FlipStreamChat_is_closed = True

    def chat(self, system, user, stop, messages, response_format, **kwargs):
        if system != self.system:
            self.system = system
            messages.clear()
        if system and len(messages) == 0:
            messages.append(dict(role="system", content=system))
        if system and messages[0]["role"] != "system":
            messages[0] = dict(role="system", content=system)
        if user:
            messages.append(dict(role="user", content=user))
        if response_format:
            response_format = json.loads(response_format)
        else:
            response_format = None
        return self.model.create_chat_completion(messages, stop=list(filter(str.strip, stop.split(","))), response_format=response_format, **kwargs)["choices"][0]["message"]

    def run(self, model_file, n_ctx, n_gpu_layers, unload_other_models, close_after_use, system, user, instant, max_history, stop, chat_model=None, messages=None, **kwargs):
        if unload_other_models:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache(True)
            try:
                comfy.gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except:
                pass
        if chat_model is not None:
            self.model = chat_model
        if messages is None:
            messages = self.messages
        if max_history == 0:
            messages.clear()
        elif max_history >= 1:
            messages[:] = messages[:max_history]
        self.load_model(model_file, n_ctx, n_gpu_layers)
        if instant:
            res = self.chat(system, user, stop, messages.copy(), **kwargs)
            output = res["content"]
        else:
            res = self.chat(system, user, stop, messages, **kwargs)
            output = res["content"]
            if res["role"] == "assistant":
                messages.append(dict(role="assistant", content=output))
        self.messages = messages
        if close_after_use:
            self.close_model()
        return (self.model, output, messages)


class FlipStreamParseJson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_input": ("STRING", {"default": "", "multiline": True}),
                "keys": ("STRING", {"default": "", "multiline": True}),
                "joinstr": ("STRING", {"default": ",", "multiline": True}),
                "ignore_error": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, json_input, keys, joinstr, ignore_error):
        value = []
        try:
            for key in keys.split("\n"):
                value.append(json.loads(json_input, strict=False)[key.strip()])
        except Exception as e:
            if not ignore_error:
                raise RuntimeError(f"FlipStreamParseJsonItem: Invalid JSON input: {e}: {json_input}")
        return (joinstr.join(value),)


class FlipStreamBatchPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "clip": ("CLIP",),
                "frames": ("INT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, prompt, clip, frames):
        cond_buf = []
        pooled_buf = []

        buf = prompt.split("----")
        prompt = buf[0].strip()
        batchPrompt = buf[1].strip() if len(buf) > 1 else ""
        appPrompt = buf[2].strip() if len(buf) > 2 else ""
        count = len(batchPrompt.split("\n"))
        batchPrompt = ",\n".join([f'"{int(frames * n / count)}":"{item.lstrip("-").strip()}"' for n, item in enumerate(batchPrompt.split("\n"))])

        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        s = json.loads("{" + batchPrompt + "}")
        for i in range(frames):
            if str(i) in s:
                tokens = clip.tokenize(" ".join([prompt, s[str(i)], appPrompt]).strip())
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_buf.append(cond)
            pooled_buf.append(pooled)
        
        max_length = max([t.size(1) for t in cond_buf])
        cond_buf = [
            NNF.pad(t, (0, 0, 0, max_length - t.size(1))) if t.size(1) < max_length else t[:, :max_length, :]
            for t in cond_buf
        ]
        return ([[torch.cat(cond_buf, dim=0), {"pooled_output":torch.cat(pooled_buf, dim=0)}]],)


class FlipStreamFilmVfi:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "multiplier": ("INT", {"default": 1, "min": 1, "max": 16}),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"        

    model = None

    def run(self, frames, multiplier, clear_cache_after_n_frames):
        if frames.shape[0] < 2 or multiplier < 2:
            return (frames,)
        
        if film is None:
            raise RuntimeError("FlipStreamFilmVfi: ComfyUI-Frame-Interpolation must be installed to use this function.")
        
        if FlipStreamFilmVfi.model is None:
            model_path = film.load_file_from_github_release("film", "film_net_fp32.pt")
            model = torch.jit.load(model_path, map_location="cpu")
            model.eval()
            FlipStreamFilmVfi.model = model
        model = FlipStreamFilmVfi.model
        model = model.to(comfy.model_management.get_torch_device())
        dtype = torch.float32

        frames = film.preprocess_frames(frames)
        number_of_frames_processed_since_last_cleared_cuda_cache = 0
        output_frames = []
        
        for frame_itr in range(len(frames) - 1):
            frame_0 = frames[frame_itr:frame_itr+1].to(comfy.model_management.get_torch_device()).float()
            frame_1 = frames[frame_itr+1:frame_itr+2].to(comfy.model_management.get_torch_device()).float()
            relust = film.inference(model, frame_0, frame_1, multiplier - 1)
            output_frames.extend([frame.detach().cpu().to(dtype=dtype) for frame in relust[:-1]])

            number_of_frames_processed_since_last_cleared_cuda_cache += 1
            if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                film.soft_empty_cache()
                number_of_frames_processed_since_last_cleared_cuda_cache = 0

        output_frames.append(frames[-1:].to(dtype=dtype))
        output_frames = [frame.cpu() for frame in output_frames]
        out = torch.cat(output_frames, dim=0)
        film.soft_empty_cache()
        model.to(torch.device("cpu"))
        return (film.postprocess_frames(out),)


class FlipStreamViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("IMAGE",),
                "allowip": ("STRING", {"default": ""}),
                "idle": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 30}),
                "loramode": ("STRING", {"default": ""}),
                "reset_updating": ("BOOLEAN", {"default": True}),
                "pingpong": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, allowip, idle, loramode, **kwargs):
        global allowed_ips
        allowed_ips = ["127.0.0.1"] + list(map(str.strip, allowip.split(",")))
        state["loraMode"] = loramode
        time.sleep(idle)
        return None

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, tensor, fps, reset_updating, pingpong, **kwargs):
        fb = []
        buf = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        if tensor.shape[0] != 1 and pingpong:
            buf = np.concatenate([buf, np.flip(buf, axis=0)])
        for image in buf:
            with io.BytesIO() as output:
                img = Image.fromarray(image)
                img.save(output, format="PNG", compress_level=STREAM_COMPRESSION)
                fb.append(output.getvalue())
        global frame_buffer
        global frame_updating
        global frame_mtime
        global frame_fps
        frame_buffer = fb
        if reset_updating:
            frame_updating = None
        frame_mtime = time.time()
        frame_fps = fps
        return ()


NODE_CLASS_MAPPINGS = {
    "FlipStreamSection": FlipStreamSection,
    "FlipStreamButton": FlipStreamButton,
    "FlipStreamSlider": FlipStreamSlider,
    "FlipStreamTextBox": FlipStreamTextBox,
    "FlipStreamInputBox": FlipStreamInputBox,
    "FlipStreamSelectBox_Samplers": FlipStreamSelectBox_Samplers,
    "FlipStreamSelectBox_Scheduler": FlipStreamSelectBox_Scheduler,
    "FlipStreamFileSelect_Checkpoints": FlipStreamFileSelect_Checkpoints,
    "FlipStreamFileSelect_Loras": FlipStreamFileSelect_Loras,
    "FlipStreamFileSelect_VAE": FlipStreamFileSelect_VAE,
    "FlipStreamFileSelect_ControlNetModel": FlipStreamFileSelect_ControlNetModel,
    "FlipStreamFileSelect_TensorRT": FlipStreamFileSelect_TensorRT,
    "FlipStreamFileSelect_AnimateDiffModel": FlipStreamFileSelect_AnimateDiffModel,
    "FlipStreamFileSelect_Input": FlipStreamFileSelect_Input,
    "FlipStreamFileSelect_Output": FlipStreamFileSelect_Output,
    "FlipStreamPreviewBox": FlipStreamPreviewBox,
    "FlipStreamLogBox": FlipStreamLogBox,
    "FlipStreamSetUpdateAndReload": FlipStreamSetUpdateAndReload,
    "FlipStreamSetMessage": FlipStreamSetMessage,
    "FlipStreamSetParam": FlipStreamSetParam,
    "FlipStreamGetParam": FlipStreamGetParam,
    "FlipStreamGetFrame": FlipStreamGetFrame,
    "FlipStreamGetPreviewRoi": FlipStreamGetPreviewRoi,
    "FlipStreamImageSize": FlipStreamImageSize,
    "FlipStreamTextReplace": FlipStreamTextReplace,
    "FlipStreamScreenGrabber": FlipStreamScreenGrabber,
    "FlipStreamVideoInput": FlipStreamVideoInput,
    "FlipStreamSource": FlipStreamSource,
    "FlipStreamSwitch": FlipStreamSwitch,
    "FlipStreamSwitchImage": FlipStreamSwitchImage,
    "FlipStreamSwitchLatent": FlipStreamSwitchLatent,
    "FlipStreamGate": FlipStreamGate,
    "FlipStreamRembg": FlipStreamRembg,
    "FlipStreamSegMask": FlipStreamSegMask,
    "FlipStreamChat": FlipStreamChat,
    "FlipStreamParseJson": FlipStreamParseJson,
    "FlipStreamBatchPrompt": FlipStreamBatchPrompt,
    "FlipStreamFilmVfi": FlipStreamFilmVfi,
    "FlipStreamViewer": FlipStreamViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlipStreamSection": "FlipStreamSection",
    "FlipStreamButton": "FlipStreamButton",
    "FlipStreamSlider": "FlipStreamSlider",
    "FlipStreamTextBox": "FlipStreamTextBox",
    "FlipStreamInputBox": "FlipStreamInputBox",
    "FlipStreamSelectBox_Samplers": "FlipStreamSelectBox_Samplers",
    "FlipStreamSelectBox_Scheduler": "FlipStreamSelectBox_Scheduler",
    "FlipStreamFileSelect_Checkpoints": "FlipStreamFileSelect_Checkpoints",
    "FlipStreamFileSelect_Loras": "FlipStreamFileSelect_Loras",
    "FlipStreamFileSelect_VAE": "FlipStreamFileSelect_VAE",
    "FlipStreamFileSelect_ControlNetModel": "FlipStreamFileSelect_ControlNetModel",
    "FlipStreamFileSelect_TensorRT": "FlipStreamFileSelect_TensorRT",
    "FlipStreamFileSelect_AnimateDiffModel": "FlipStreamFileSelect_AnimateDiffModel",
    "FlipStreamFileSelect_Input": "FlipStreamFileSelect_Input",
    "FlipStreamFileSelect_Output": "FlipStreamFileSelect_Output",
    "FlipStreamPreviewBox": "FlipStreamPreviewBox",
    "FlipStreamLogBox": "FlipStreamLogBox",
    "FlipStreamSetUpdateAndReload": "FlipStreamSetUpdateAndReload",
    "FlipStreamSetMessage": "FlipStreamSetMessage",
    "FlipStreamSetParam": "FlipStreamSetParam",
    "FlipStreamGetParam": "FlipStreamGetParam",
    "FlipStreamGetFrame": "FlipStreamGetFrame",
    "FlipStreamGetPreviewRoi": "FlipStreamGetPreviewRoi",
    "FlipStreamImageSize": "FlipStreamImageSize",
    "FlipStreamTextReplace": "FlipStreamTextReplace",
    "FlipStreamScreenGrabber": "FlipStreamScreenGrabber",
    "FlipStreamVideoInput": "FlipStreamVideoInput",
    "FlipStreamSource": "FlipStreamSource",
    "FlipStreamSwitch": "FlipStreamSwitch",
    "FlipStreamSwitchImage": "FlipStreamSwitchImage",
    "FlipStreamSwitchLatent": "FlipStreamSwitchLatent",
    "FlipStreamGate": "FlipStreamGate",
    "FlipStreamRembg": "FlipStreamRembg",
    "FlipStreamSegMask": "FlipStreamSegMask",
    "FlipStreamChat": "FlipStreamChat",
    "FlipStreamParseJson": "FlipStreamParseJson",
    "FlipStreamBatchPrompt": "FlipStreamBatchPrompt",
    "FlipStreamFilmVfi": "FlipStreamFilmVfi",
    "FlipStreamViewer": "FlipStreamViewer",
}
