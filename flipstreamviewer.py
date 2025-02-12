import asyncio
import base64
import hashlib
import html
import json
import threading
import time
from pathlib import Path
from io import BytesIO

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

try:
    wd14tagger = __import__("comfyui-wd14-tagger").wd14tagger
except:
    wd14tagger = None

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
    from custom_nodes.ComfyUI_TensorRT import TensorRTLoader
except:
    TensorRTLoader = None

def btoa(value):
    return base64.b64encode(value.encode("latin-1")).decode()

def atob(value):
    return base64.b64decode(value.encode()).decode("latin-1")

STREAM_COMPRESSION = 1
allowed_ips = ["127.0.0.1"]
setparam = {}
param = {"loramode": "", "lora": "", "_capture_offsetX": 0, "_capture_offsetY": 0, "_capture_scale": 100}
state = {"presetTitle": time.strftime("%Y%m%d-%H%M"), "presetFolder": "", "presetFile": "", "loraRate": "1", "loraRank": "0", "loraFolder": "", "loraFile": "", "loraTagOptions": "[]", "loraTag": "", "loraLinkHref": "", "loraPreviewSrc": "", "darker": 0.0, "wd14th": 0.35, "wd14cth": 0.85}
frame_updating = False
frame_buffer = None
frame_mtime = 0
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
    background: url('/flipstreamviewer/stream') rgba(0, 0, 0, 0) no-repeat top center local;
    background-blend-mode: overlay;
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

#statusInfo {
    line-break: anywhere;
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

#presetFileSelect,
#presetTitleInput,
#loraFileSelect,
#loraTagSelect {
    width: 80%;
}

#darkerRange,
#wd14thRange,
#wd14cthRange {
    width: 50%;
}

#toggleViewButton {
    width: 100%;
    height: 100%;
    border: none;
    background: transparent;
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
    const wd14th = parseFloat(document.getElementById("wd14thRange").value);
    const wd14cth = parseFloat(document.getElementById("wd14cthRange").value);
    var res = { presetTitle: presetTitle, presetFolder: presetFolder, presetFile: presetFile, loraRate: loraRate, loraRank: loraRank, loraFolder: loraFolder, loraFile: loraFile, loraTagOptions: loraTagOptions, loraTag: loraTag, loraLinkHref: loraLinkHref, loraPreviewSrc: loraPreviewSrc, loraTagOptions: loraTagOptions, darker: darker, wd14th: wd14th, wd14cth: wd14cth };
    document.querySelectorAll('.FlipStreamFolderSelect').forEach(x => res[x.name] = x.value);
    res = Object.assign(res, force_state);
    return res;
}

function getParamAsJson(force_param={}) {
    const lora = btoa(document.getElementById("loraInput").value.trim());
    const _capture_offsetX = parseInt(document.getElementById("offsetXRange").value);
    const _capture_offsetY = parseInt(document.getElementById("offsetYRange").value);
    const _capture_scale = parseInt(document.getElementById("scaleRange").value);

    var res = { lora: lora, _capture_offsetX: _capture_offsetX, _capture_offsetY: _capture_offsetY, _capture_scale: _capture_scale }
    document.querySelectorAll('.FlipStreamSlider').forEach(x => res[x.name] = x.value);
    document.querySelectorAll('.FlipStreamTextBox').forEach(x => res[x.name] = btoa(x.value));
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
    var updateButton = document.getElementById("updateButton");
    updateButton.disabled = true;
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
    updateButton.disabled = false;
}
"""

SCRIPT_PRESET=r"""
function loadPreset(loraPromptOnly=false, force_state={}, search="") {
    fetch("/flipstreamviewer/load_preset", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(force_state), loraPromptOnly]),
        headers: {"Content-Type": "application/json"}
    }).then(response => response.json()).then(json => {
        if (loraPromptOnly) {
            document.getElementById("loraInput").value = atob(json.lora);
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
        body: JSON.stringify([getStateAsJson(), getParamAsJson()]),
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

async function addWD14Tag() {
    const json = await (await fetch("/flipstreamviewer/get_wd14tag", {
        method: "POST",
        body: JSON.stringify(getStateAsJson()),
        headers: {"Content-Type": "application/json"}
    })).json();
    if (json.tags != "") {
        loraInput.value = json.tags + "\n\n" + loraInput.value;
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
function toggleView() {
    const leftPanel = document.getElementById("leftPanel");
    const rightPanel = document.getElementById("rightPanel");
    const presetLeftPanel = document.getElementById("presetLeftPanel");
    const presetRightPanel = document.getElementById("presetRightPanel");
    if (toggleViewFlag) {
        leftPanel.style.visibility = "hidden";
        rightPanel.style.visibility = "hidden";
        presetLeftPanel.style.visibility = "hidden";
        presetRightPanel.style.visibility = "hidden";
        toggleViewFlag = false;
    } else {
        document.body.style.backgroundImage = "none";
        document.getElementById("loraPreview").src = "";
        document.querySelectorAll('.FlipStreamPreviewBox').forEach(x => x.src = "");
        leftPanel.style.visibility = "visible";
        rightPanel.style.visibility = "visible";
        presetLeftPanel.style.visibility = "visible";
        presetRightPanel.style.visibility = "visible";
        toggleViewFlag = true;
    }
}

function hideView() {
    const leftPanel = document.getElementById("leftPanel");
    const rightPanel = document.getElementById("rightPanel");
    const presetLeftPanel = document.getElementById("presetLeftPanel");
    const presetRightPanel = document.getElementById("presetRightPanel");
    closeCaptureDialog();
    document.body.style.backgroundImage = "none";
    document.querySelectorAll('.FlipStreamPreviewBox').forEach(x => x.src = "");
    leftPanel.style.visibility = "hidden";
    rightPanel.style.visibility = "hidden";
    presetLeftPanel.style.visibility = "hidden";
    presetRightPanel.style.visibility = "hidden";
    toggleViewFlag = false;
}
setTimeout(hideView, 300 * 1000);

async function refreshView() {
    document.getElementById("statusInfo").innerText = await fetch("/flipstreamviewer/get_status")
        .then(response => response.text())
        .catch(() => "Fails to get status.");

    if (document.body.style.backgroundImage != "none") {
        const mtime = await (await fetch("/flipstreamviewer/stream_mtime")).text();
        document.body.style.backgroundImage = `url('/flipstreamviewer/stream?mtime=${mtime}')`;
    }

    document.querySelectorAll('.FlipStreamPreviewBox').forEach(async x => {
        if (x.src != "") {
            const mtime = await (await fetch(`/flipstreamviewer/preview_mtime?label=${x.name}`)).text();
            x.src = `/flipstreamviewer/preview?label=${x.name}&mtime=${mtime}`;
        }
    });
}
setInterval(refreshView, 1000);
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
function onInputDarker(value) {
    value = parseFloat(value);
    document.getElementById("darkerRange").value = value;
    document.getElementById("darkerValue").innerText = value; 
    document.body.style.backgroundColor = 'rgba(0,0,0,' + value + ')';
}

function reloadPage(search="") {
    location.replace(location.pathname + search && "?" + search);
}

function parseQueryParam() {
    const p = new URLSearchParams(location.search)
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

@server.PromptServer.instance.routes.get("/flipstreamviewer/stream")
async def stream(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    return web.Response(body=frame_buffer, headers={"Content-Type": "image/apng"})


@server.PromptServer.instance.routes.get("/flipstreamviewer/stream_mtime")
async def stream_mtime(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    return web.Response(text=str(frame_mtime))


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


@server.PromptServer.instance.routes.get("/flipstreamviewer/preview_mtime")
async def preview_mtime(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    label = request.rel_url.query.get('label', '')
    key = label + "PreviewBox"
    if key in state:
        data = state[key][0]
    else:
        data = 0

    return web.Response(text=str(data))


@server.PromptServer.instance.routes.get("/flipstreamviewer")
async def viewer(request):
    if request.remote not in allowed_ips:
        print(request.remote)
        raise HTTPForbidden()

    block = {}

    def add_section(title, section, hook):
        block[f"{title}_{section}"] = f"""
            <div class="row"><i>{section}</i></div>"""

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
        param.setdefault(label, default)
        block[f"{title}_{label}"] = f"""
            <textarea class="FlipStreamTextBox" id="{label}TextBox" style="color: lightslategray;" placeholder="{label}" rows="{rows}" name="{label}">{atob(param[label])}</textarea>"""

    def add_inputbox(title, label, default, boxtype):
        if not label.isidentifier():
            raise RuntimeError(f"{title}: label must contain only valid identifier characters.")
        param.setdefault(label, default)
        if boxtype == "seed":
            block[f"{title}_{label}"] = f"""
            <div class="row" style="color: lightslategray;">
                {label}: <input class="FlipStreamInputBox" id="{label}InputBox" style="color: lightslategray;" placeholder="{label}" type="number" name="{label}" value="{param[label]}" />
                <button onclick="{label}InputBox.value=Math.floor(Math.random()*1e7); updateParam()">R</button>
            </div>"""
        elif boxtype == "r4d":
            block[f"{title}_{label}"] = f"""
            <div class="row" style="color: lightslategray;">
                {label}: <input class="FlipStreamInputBox" id="{label}InputBox" style="color: lightslategray;" placeholder="{label}" type="number" name="{label}" value="{param[label]}" />
                <button onclick="{label}InputBox.value=Math.floor(Math.random()*1e4); updateParam()">R</button>
            </div>"""
        else:
            block[f"{title}_{label}"] = f"""
            <div class="row" style="color: lightslategray;">
                {label}: <input class="FlipStreamInputBox" id="{label}InputBox" style="color: lightslategray;" placeholder="{label}" type="{boxtype}" name="{label}" value="{param[label]}" />
                <button onclick="updateParam()">U</button>
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

    def add_fileselect(title, label, default, folder_name, folder_path, mode, use_lora, use_sub, use_move):
        if not label.isidentifier():
            raise RuntimeError(f"{title}: label must contain only valid identifier characters.")
        if not (mode == "" or mode.isidentifier()):
            raise RuntimeError(f"{title}: mode must contain only valid identifier characters.")
        param.setdefault(label, "")
        state.setdefault(f"{label}Folder", "")
        if use_lora:
            param["loramode"] = mode
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
            files = [Path(file) for file in FlipStreamFileSelect.get_filelist(folder_name, folder_path)]
        
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
            </div>"""
    
    hist = server.PromptServer.instance.prompt_queue.get_history(max_items=1)
    nodedict = next(iter(hist.values()))["prompt"][2] if hist else None
    if nodedict:
        for node in nodedict.values():
            class_type = node["class_type"]
            title = node["_meta"]["title"]
            inputs = node["inputs"]
            if class_type == "FlipStreamSection":
                add_section(title, **inputs)
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

    text_html = f"""<html>{HEAD}<body>
    <div id="mainDialog">
        <div id="leftPanel">
            <div class="row">
                <button id="updateButton" class="willreload" onclick="updateParam(true)">Update and reload</button>
            </div>
            {"".join([x[1] for x in sorted(block.items())])}
        </div>
        <div id="centerPanel">
            <button id="toggleViewButton" onclick="toggleView()"></button>
        </div>
        <div id="rightPanel">
            <div class="row"><i>Status</i></div>
            <div id="statusInfo"></div>
            <div class="row"><i>Darker</i></div>
            <div class="row">
                <input id="darkerRange" type="range" min="0" max="1" step="0.01" value="{state["darker"]}" oninput="onInputDarker(this.value);" />
                <span id="darkerValue">{state["darker"]}</span>drk
            </div>
            <div class="row"><i>Tagger</i></div>
            <div class="row">
                <button onclick="capture()">Capture</button>
                <button onclick="addWD14Tag()">WD14</button>
            </div>
            <div class="row">
                <input id="wd14thRange" type="range" min="0" max="1" step="0.01" value="{state["wd14th"]}" oninput="wd14thValue.innerText = this.value;" />
                <span id="wd14thValue">{state["wd14th"]}</span>wth
            </div>
            <div class="row">
                <input id="wd14cthRange" type="range" min="0" max="1" step="0.01" value="{state["wd14cth"]}" oninput="wd14cthValue.innerText = this.value;" />
                <span id="wd14cthValue">{state["wd14cth"]}</span>cth
            </div>
            <div class="row"><i>Preset</i></div>            
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
            <div class="row">
                <button onclick="showPresetDialog()">Choose</button>
                <button class="willreload" onclick="loadPreset()">Load</button>
                <button class="willreload" onclick="loadPreset(true)">LoraOnly</button>
            </div>
            <div class="row"><i>Lora</i></div>
            <select id="loraFolderSelect" class="willreload" onchange="updateParam(true)">
                <option value="" selected>lora folder</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["loraFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("ComfyUI/models/loras", param["loramode"]).glob("*/")])}
            </select>
            <div class="row">
                <select id="loraFileSelect" onchange="selectLoraFile()">
                    <option value="" disabled selected>lora file</option>
                    {"".join([f'<option value="{file.name}"{" selected" if state["loraFile"] == file.name else ""}>{file.stem}</option>' for file in Path("ComfyUI/models/loras", param["loramode"], state["loraFolder"]).glob("*.safetensors")])}
                </select>
                <button onclick="toggleLora()">T</button>
                <button onClick="moveLora()">M</button>
            </div>
            <select id="moveLoraSelect" onchange="moveLoraFile()">
                <option value="" disabled selected>move to</option>
                {"".join([f'<option value="{dir.name}">{dir.name}</option>' for dir in Path("ComfyUI/models/loras", param["loramode"]).glob("*/")])}
            </select>
            <div class="row">
                <select id="loraTagSelect" onchange="toggleTag()">
                    <option value="" disabled selected>tags</option>
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
                <button onclick="updateParam()">Update</button>
                <button onclick="clearLoraInput()">Clr</button>
                <button onclick="showTagDialog();tagCSel();tagRSel();tagOK();updateParam()">R</button>
            </div>
            <textarea id="loraInput" placeholder="Enter lora" rows="12">{atob(param["lora"])}</textarea>
            <div class="row">
                <a id="loraLink" href="{state["loraLinkHref"] or "javascript:void(0)"}" target="_blank">
                    <img id="loraPreview" src="{state["loraPreviewSrc"]}" alt onerror="this.onerror = null; this.src='';" />
                </a>
            </div>
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
        <div id="presetCenterPanel">
            <button id="toggleViewButton" onclick="toggleView()"></button>
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
            {"".join([f"""
            <button onclick="loadPreset(false, {{ presetFile: '{file.name}' }}, 'showPresetDialog')">{file.stem}</button>
            """ for file in Path("preset", state["presetFolder"]).glob("*.json")])}
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


@server.PromptServer.instance.routes.get("/flipstreamviewer/get_status")
async def get_status(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    remain = server.PromptServer.instance.prompt_queue.get_tasks_remaining()
    hist = server.PromptServer.instance.prompt_queue.get_history(max_items=1)
    status = []
    if frame_updating:
        status.append("updating")
    status.append(f"q{remain}")
    info = next(iter(hist.values()))["status"] if hist else None
    if info:
        errinfo = info["messages"][2][1]
        status.append(info["status_str"])
        status += [errinfo.get(key, "") for key in ["node_id", "node_type", "exception_message", "exception_type"]]
    return web.Response(text=":".join(status))


@server.PromptServer.instance.routes.post("/flipstreamviewer/update_param")
async def update_param(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, prm = await request.json()
    state.update(stt)
    param.update(prm)
    param.update(setparam)
    setparam.clear()
    return web.Response()


@server.PromptServer.instance.routes.post("/flipstreamviewer/get_lorainfo")
async def get_lorainfo(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    req = await request.json()
    loraPath = Path("ComfyUI/models/loras", param["loramode"], req["loraFolder"], req["loraFile"])
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


@server.PromptServer.instance.routes.post("/flipstreamviewer/get_wd14tag")
async def get_wd14tag(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    if wd14tagger is None:
        raise RuntimeError("get_wd14tag: ComfyUI-WD14-Tagger must be installed to use this function.")

    stt = await request.json()
    state.update(stt);
    tags = []
    if frame_buffer is not None:
        tags = await wd14tagger.tag(Image.fromarray(iio.imread(frame_buffer, index=0)), "wd-v1-4-moat-tagger-v2.onnx", state["wd14th"], state["wd14cth"], exclude_tags)
    return web.json_response({"tags": tags})


@server.PromptServer.instance.routes.post("/flipstreamviewer/set_frame")
async def set_frame(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()
    
    global frame_buffer
    global frame_mtime
    pngdata = base64.b64decode((await request.text()).split(',', 1)[1])
    with BytesIO(pngdata) as data:
        with BytesIO() as output:
            iio.imwrite(output, [iio.imread(data)], format="png", compression=STREAM_COMPRESSION)
            frame_buffer = output.getvalue()
            frame_mtime = time.time()
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
    pathFrom = Path("ComfyUI/models/loras", param["loramode"], state["loraFolder"], state["loraFile"])
    pathTo = Path("ComfyUI/models/loras", param["loramode"], moveTo, pathFrom.name)
    Path(pathFrom).rename(pathTo)
    state["loraFolder"] = moveTo
    return web.Response()


@server.PromptServer.instance.routes.post("/flipstreamviewer/load_preset")
async def load_preset(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, loraPromptOnly = await request.json()
    state.update(stt)
    data = param
    with open(Path("preset", state["presetFolder"], state["presetFile"]), "r") as file:
        buf = json.load(file)
        if loraPromptOnly:
            if "lora" in buf:
                data["lora"] = buf["lora"]
        else:
            data = buf

    param.update(data)
    state["presetTitle"] = Path(state["presetFile"]).stem
    return web.json_response({"lora": data["lora"], "presetTitle": state["presetTitle"]})


@server.PromptServer.instance.routes.post("/flipstreamviewer/save_preset")
async def save_preset(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    [stt, data] = await request.json()
    state.update(stt)
    param.update(data)
    with open(Path("preset", state["presetFolder"], state["presetTitle"] + ".json"), "w") as file:
        json.dump(data, file)
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
        frame_updating = True
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
        param.setdefault(label, btoa(default))
        return hash((param[label],))
    
    def run(self, label, default, **kwargs):
        global frame_updating
        frame_updating = True
        param.setdefault(label, btoa(default))        
        return (atob(param[label]),)


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
        frame_updating = True
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
        frame_updating = True
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
    def get_filelist(folder_name, folder_path):
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
            return list(map(str, Path(folder_path).glob("*.*")))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label": ("STRING", {"default": "empty"}),
                "default": ([""] + s.get_filelist(s.FOLDER_NAME, s.FOLDER_PATH),),
                "folder_name": ([s.FOLDER_NAME],),
                "folder_path": ([s.FOLDER_PATH],),
                "mode": ("STRING", {"default": ""}),
                "use_lora": ("BOOLEAN", {"defalut": False}),
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

    def run(self, label, default, folder_name, folder_path, **kwargs):
        global frame_updating
        frame_updating = True
        param.setdefault(label, "")
        file = param[label] if param[label] else default
        return (file, str(Path(folder_path, file)), param[label] != "")


class FlipStreamFileSelect_Checkpoints(FlipStreamFileSelect):
    FOLDER_NAME = "checkpoints"
    FOLDER_PATH = str(Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()))


class FlipStreamFileSelect_VAE(FlipStreamFileSelect):
    FOLDER_NAME = "vae"
    FOLDER_PATH = str(Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()))


class FlipStreamFileSelect_ControlNetModel(FlipStreamFileSelect):
    FOLDER_NAME = "controlnet"
    FOLDER_PATH = str(Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()))


class FlipStreamFileSelect_TensorRT(FlipStreamFileSelect):
    FOLDER_NAME = "tensorrt"
    try:
        FOLDER_PATH = str(Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()))
    except:
        FOLDER_PATH = "_error_ tensorrt folder is not found"


class FlipStreamFileSelect_AnimateDiffModel(FlipStreamFileSelect):
    FOLDER_NAME = "animatediff_models"
    try:
        FOLDER_PATH = str(Path(folder_paths.get_folder_paths(FOLDER_NAME)[0]).relative_to(Path.cwd()))
    except:
        FOLDER_PATH = "_error_ animatediff_models folder is not found"


class FlipStreamFileSelect_Input(FlipStreamFileSelect):
    FOLDER_NAME = "input"
    FOLDER_PATH = str(Path(folder_paths.input_directory).relative_to(Path.cwd()))


class FlipStreamFileSelect_Output(FlipStreamFileSelect):
    FOLDER_NAME = "output"
    FOLDER_PATH = str(Path(folder_paths.output_directory).relative_to(Path.cwd()))


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
        buf = np.array(tensor[0].cpu().numpy() * 255, dtype=np.uint8)
        image = Image.fromarray(buf)
        image.thumbnail((256, 256))
        with BytesIO() as output:
            iio.imwrite(output, np.array(image), format="png", compression=STREAM_COMPRESSION)
            state[label + "PreviewBox"] = (time.time(), output.getvalue())
        return ()


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
            value = btoa(value)
            empty = btoa(empty)
        if replace or label not in param or param[label] == empty:
            setparam[label] = value
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
                value = atob(value)
        return hash((value,))

    def run(self, label, default, b64dec):
        global frame_updating
        frame_updating = True
        value = default
        if label in param:
            value = param[label]
            if b64dec:
                value = atob(value)
        return (value,)


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
                "text": ("STRING", {"default": ""}),
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
        return (text.replace(find, replace.format(value)),)


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
            if height != buf.shape[2]:
                buf = buf.movedim(-1,1)
                buf = comfy.utils.common_upscale(buf, round(buf.shape[3] * height / buf.shape[2]), height, "lanczos", "disabled")
                buf = buf.movedim(1,-1)
            image = torch.zeros([frames, height, width, 3])
            x2 = width // 2
            w2 = buf.shape[2] // 2
            if (x2 - w2 >= 0):
                image[:, :, x2-w2:x2+w2] = buf[:, :, :w2*2]
            else:
                image = buf[:, :, w2-x2:w2+x2]
            if vae:
                latent = {"samples": vae.encode(image)}
        else:
            image = torch.zeros([frames, height, width, 3])
            if vae:
                latent = {"samples": torch.zeros([frames, 4, height // 8, width // 8], device=comfy.model_management.intermediate_device())}
        return (image, latent,)


class FlipStreamSwitchImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "image": ("IMAGE",),
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
        mm = comfy.model_management
        if FlipStreamSegMask.model is None:
            FlipStreamSegMask.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation='sdpa',
                device_map=mm.get_torch_device(),
                torch_dtype=torch.bfloat16
            )
            FlipStreamSegMask.processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = FlipStreamSegMask.model
        processor = FlipStreamSegMask.processor
        model.to(mm.get_torch_device())

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
        mm = comfy.model_management
        model = FlipStreamFilmVfi.model
        model = model.to(mm.get_torch_device())
        dtype = torch.float32

        frames = film.preprocess_frames(frames)
        number_of_frames_processed_since_last_cleared_cuda_cache = 0
        output_frames = []
        
        for frame_itr in range(len(frames) - 1):
            frame_0 = frames[frame_itr:frame_itr+1].to(mm.get_torch_device()).float()
            frame_1 = frames[frame_itr+1:frame_itr+2].to(mm.get_torch_device()).float()
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
                "wd14exc": ("STRING", {"default": ""}),
                "idle": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "fps": ("INT", {"default": 8, "min": 1, "max": 30}),
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
    FUNCTION = "run"
    CATEGORY = "FlipStreamViewer"

    def run(self, tensor, fps, **kwargs):
        global frame_updating
        global frame_buffer
        global frame_mtime
        buf = np.array(tensor.cpu().numpy() * 255, dtype=np.uint8)
        with BytesIO() as output:
            iio.imwrite(output, list(buf) + list(buf[::-1]), format='png', compression=STREAM_COMPRESSION, fps=fps)
            frame_buffer = output.getvalue()
            frame_mtime = time.time()
        frame_updating = False
        return ()


NODE_CLASS_MAPPINGS = {
    "FlipStreamSection": FlipStreamSection,
    "FlipStreamSlider": FlipStreamSlider,
    "FlipStreamTextBox": FlipStreamTextBox,
    "FlipStreamInputBox": FlipStreamInputBox,
    "FlipStreamSelectBox_Samplers": FlipStreamSelectBox_Samplers,
    "FlipStreamSelectBox_Scheduler": FlipStreamSelectBox_Scheduler,
    "FlipStreamFileSelect_Checkpoints": FlipStreamFileSelect_Checkpoints,
    "FlipStreamFileSelect_VAE": FlipStreamFileSelect_VAE,
    "FlipStreamFileSelect_ControlNetModel": FlipStreamFileSelect_ControlNetModel,
    "FlipStreamFileSelect_TensorRT": FlipStreamFileSelect_TensorRT,
    "FlipStreamFileSelect_AnimateDiffModel": FlipStreamFileSelect_AnimateDiffModel,
    "FlipStreamFileSelect_Input": FlipStreamFileSelect_Input,
    "FlipStreamFileSelect_Output": FlipStreamFileSelect_Output,
    "FlipStreamPreviewBox": FlipStreamPreviewBox,
    "FlipStreamSetParam": FlipStreamSetParam,
    "FlipStreamGetParam": FlipStreamGetParam,
    "FlipStreamImageSize": FlipStreamImageSize,
    "FlipStreamTextReplace": FlipStreamTextReplace,
    "FlipStreamScreenGrabber": FlipStreamScreenGrabber,
    "FlipStreamSource": FlipStreamSource,
    "FlipStreamSwitchImage": FlipStreamSwitchImage,
    "FlipStreamSwitchLatent": FlipStreamSwitchLatent,
    "FlipStreamRembg": FlipStreamRembg,
    "FlipStreamSegMask": FlipStreamSegMask,
    "FlipStreamBatchPrompt": FlipStreamBatchPrompt,
    "FlipStreamFilmVfi": FlipStreamFilmVfi,
    "FlipStreamViewer": FlipStreamViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlipStreamSection": "FlipStreamSection",
    "FlipStreamSlider": "FlipStreamSlider",
    "FlipStreamTextBox": "FlipStreamTextBox",
    "FlipStreamInputBox": "FlipStreamInputBox",
    "FlipStreamSelectBox_Samplers": "FlipStreamSelectBox_Samplers",
    "FlipStreamSelectBox_Scheduler": "FlipStreamSelectBox_Scheduler",
    "FlipStreamFileSelect_Checkpoints": "FlipStreamFileSelect_Checkpoints",
    "FlipStreamFileSelect_VAE": "FlipStreamFileSelect_VAE",
    "FlipStreamFileSelect_ControlNetModel": "FlipStreamFileSelect_ControlNetModel",
    "FlipStreamFileSelect_TensorRT": "FlipStreamFileSelect_TensorRT",
    "FlipStreamFileSelect_AnimateDiffModel": "FlipStreamFileSelect_AnimateDiffModel",
    "FlipStreamFileSelect_Input": "FlipStreamFileSelect_Input",
    "FlipStreamFileSelect_Output": "FlipStreamFileSelect_Output",
    "FlipStreamPreviewBox": "FlipStreamPreviewBox",
    "FlipStreamSetParam": "FlipStreamSetParam",
    "FlipStreamGetParam": "FlipStreamGetParam",
    "FlipStreamImageSize": "FlipStreamImageSize",
    "FlipStreamTextReplace": "FlipStreamTextReplace",
    "FlipStreamScreenGrabber": "FlipStreamScreenGrabber",
    "FlipStreamSource": "FlipStreamSource",
    "FlipStreamSwitchImage": "FlipStreamSwitchImage",
    "FlipStreamSwitchLatent": "FlipStreamSwitchLatent",
    "FlipStreamRembg": "FlipStreamRembg",
    "FlipStreamSegMask": "FlipStreamSegMask",
    "FlipStreamBatchPrompt": "FlipStreamBatchPrompt",
    "FlipStreamFilmVfi": "FlipStreamFilmVfi",
    "FlipStreamViewer": "FlipStreamViewer",
}
