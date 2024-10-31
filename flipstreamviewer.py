import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path
from io import BytesIO

import cv2
import requests
import torch
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
param = {"prompt": "", "negativePrompt": "", "seed": 0, "keepSeed": 0, "steps": 13, "cfg": 4, "interval": 30, "sampler": "dpmpp_2m,sgm_uniform", "checkpoint": "", "lora": "", "startstep": 0, "frames": 1, "framewait": 1, "videosrc": "", "videofst": 0, "videomax": 0, "videoskp": 8, "videostr": 1, "offsetX": 0, "offsetY": 0, "scale": 100}
state = {"mode": "", "height": 512, "autoUpdate": False, "presetTitle": time.strftime("%Y%m%d-%H%M"), "presetFolder": "", "presetFile": "", "loraRate": "1", "loraRank": "0", "checkpointFolder": "", "loraFolder": "", "loraFile": "", "loraTagOptions": "[]", "loraTag": "", "loraLinkHref": "", "loraPreviewSrc": "", "wd14th": 0.35, "wd14cth": 0.85}
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
    font-size: 120%;
    padding: 2px;
    min-width: 2em;
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

#leftPanel, #captureLeftPanel, #tagLeftPanel,
#rightPanel, #captureRightPanel, #tagRightPanel {
    width: 12%;
}

#centerPanel, #captureCenterPanel, #tagCenterPanel {
    width: 76%;
    text-align: center;
}

#captureDialog, #tagDialog {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    z-index: 999;
}

#samplerSelect,
#checkpointFolderSelect,
#moveCheckpointSelect,
#presetFolderSelect,
#presetFileSelect,
#movePresetSelect,
#loraFolderSelect,
#moveLoraSelect {
    width: 100%;
}

#moveCheckpointSelect,
#movePresetSelect,
#moveLoraSelect {
    display: none;
}

#seedInput,
#keepSeedInput,
#checkpointFileSelect,
#presetTitleInput,
#loraFileSelect,
#loraTagSelect,
#videoSrcSelect {
    width: 80%;
}

#stepsRange,
#cfgRange,
#startstepRange,
#framesRange,
#framewaitRange,
#videoStrRange,
#videoFstRange,
#videoSkpRange,
#wd14thRange,
#wd14cthRange {
    width: 50%;
}

#toggleView {
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
    const loraRank = document.getElementById("loraRank").value;
    const loraLinkHref = document.getElementById("loraLink").getAttribute("href");
    const loraPreviewSrc = document.getElementById("loraPreview").getAttribute("src");
    const wd14th = parseFloat(document.getElementById("wd14thRange").value);
    const wd14cth = parseFloat(document.getElementById("wd14cthRange").value);
    return { autoUpdate: autoUpdate, presetTitle: presetTitle, presetFolder: presetFolder, presetFile: presetFile, checkpointFolder: checkpointFolder, loraRate: loraRate, loraRank: loraRank, loraFolder: loraFolder, loraFile: loraFile, loraTagOptions: loraTagOptions, loraTag: loraTag, loraLinkHref: loraLinkHref, loraPreviewSrc: loraPreviewSrc, loraTagOptions: loraTagOptions, wd14th: wd14th, wd14cth: wd14cth };
}

function getParamAsJson() {
    const prompt = document.getElementById("promptInput").value.trim();
    const negativePrompt = document.getElementById("negativePromptInput").value.trim();
    const seed = parseInt(document.getElementById("seedInput").value) || 0;
    const keepSeed = parseInt(document.getElementById("keepSeedInput").value) || 0;
    const steps = parseInt(document.getElementById("stepsRange").value);
    const cfg = parseFloat(document.getElementById("cfgRange").value);
    const interval = parseInt(document.getElementById("intervalSelect").value);
    const sampler = document.getElementById("samplerSelect").value;
    const checkpoint = document.getElementById("checkpointFileSelect").value;
    const lora = document.getElementById("loraInput").value.trim();
    const startstep =  parseInt(document.getElementById("startstepRange").value);
    const frames = parseInt(document.getElementById("framesRange").value);
    const framewait = parseFloat(document.getElementById("framewaitRange").value);
    const videosrc = document.getElementById("videoSrcSelect").value;
    const videofst = parseInt(document.getElementById("videoFstRange").value);
    const videomax = parseInt(document.getElementById("videoFstRange").getAttribute("max"));
    const videoskp = parseInt(document.getElementById("videoSkpRange").value);
    const videostr = parseFloat(document.getElementById("videoStrRange").value);
    const offsetX = parseInt(document.getElementById("offsetXRange").value);
    const offsetY = parseInt(document.getElementById("offsetYRange").value);
    const scale = parseInt(document.getElementById("scaleRange").value);
    return { prompt: prompt, negativePrompt: negativePrompt, seed: seed, keepSeed: keepSeed, steps: steps, cfg: cfg, interval: interval, sampler: sampler, checkpoint: checkpoint, lora: lora, startstep: startstep, frames: frames, framewait: framewait, videosrc: videosrc, videofst: videofst, videomax: videomax, videoskp: videoskp, videostr: videostr, offsetX: offsetX, offsetY: offsetY, scale: scale };
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
    document.getElementById("videoSrcSelect").value = param.videosrc || "";
    document.getElementById("videoFstRange").setAttribute("max", param.videomax || 0);
    document.getElementById("videoFstRange").value = param.videofst || 0;
    document.getElementById("videoFstValue").innerText = param.videofst || 0;
    document.getElementById("videoSkpRange").value = param.videoskp || 8;
    document.getElementById("videoSkpValue").innerText = param.videoskp || 8;
    document.getElementById("videoStrRange").value = param.videostr || 1;
    document.getElementById("videoStrValue").innerText = param.videostr || 1;
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
function loadPreset(loraPromptOnly=false) {
    fetch("/load_preset", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(), loraPromptOnly]),
        headers: {"Content-Type": "application/json"}
    }).then(response => response.json()).then(json => {
        if (loraPromptOnly) {
            document.getElementById("loraInput").value = json.lora;
            document.getElementById("presetTitleInput").value = json.presetTitle;
        }
        else {
            location.reload();
        }
    }).catch(error => {
        alert("An error occurred while loading preset: " + error);
    });
}

function randomPreset() {
    const presetFileSelect = document.getElementById("presetFileSelect");
    presetFileSelect.selectedIndex = Math.floor(Math.random() * (presetFileSelect.options.length - 1)) + 1;
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
    fetch("/move_presetfile", {
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
    fetch("/save_preset", {
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
"""

SCRIPT_CKPT=r"""
function moveCheckpoint() {
    document.getElementById("moveCheckpointSelect").style.display = "block";
    document.getElementById("moveCheckpointSelect").value = "";
}

function moveCheckpointFile() {
    const moveFrom = document.getElementById("checkpointFileSelect").value;
    const moveTo = document.getElementById("moveCheckpointSelect").value;
    if (!moveFrom || !moveTo) {
        return;
    }
    fetch("/move_checkpointfile", {
        method: "POST",
        body: JSON.stringify([getStateAsJson(), moveFrom, moveTo]),
        headers: {"Content-Type": "application/json"}
    }).then(response => {
        if (response.ok) {
            alert("Checkpoint moved successfully!");
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
    const loraTagSelect = document.getElementById("loraTagSelect");
    const loraInput = document.getElementById("loraInput");
    const loraLink = document.getElementById("loraLink");
    const loraPreview = document.getElementById("loraPreview");
    
    if (loraFile) {
        const re = new RegExp("[\\n]?<lora:" + loraName + ":[-]?[0-9.]+>", "g");
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
    fetch("/move_lorafile", {
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

SCRIPT_VID=r"""
function selectVideoSrc() {
    const videoFstRange = document.getElementById("videoFstRange");
    const videoFstValue = document.getElementById("videoFstValue");
    videoFstRange.setAttribute("max", 0);
    videoFstRange.value = 0;
    videoFstValue.innerText = 0;
}

function previewVideo() {
    const videoFstRange = document.getElementById("videoFstRange");
    const videoFstValue = document.getElementById("videoFstValue");
    const videosrc = document.getElementById("videoSrcSelect").value;
    const videofst = parseInt(document.getElementById("videoFstRange").value);
    const videoskp = parseInt(document.getElementById("videoSkpRange").value);
    const frames = parseInt(document.getElementById("framesRange").value);
    if (!videosrc) {
        videoFstRange.setAttribute("max", 0);
        videoFstRange.value = 0;
        videoFstValue.innerText = 0;
        return;
    }
    fetch("/preview_video", {
        method: "POST",
        body: JSON.stringify([videosrc, videofst, videoskp, frames]),
        headers: {"Content-Type": "application/json"}
    }).then(response => response.json()).then(json => {
        videoFstRange.setAttribute("max", json.total_frames - 1);
        videoFstRange.value = Math.min(videofst, json.total_frames - 1);
        videoFstValue.innerText = videoFstRange.value;
    }).catch(error => {
        alert("An error occurred while preview video.");
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
    fetch("/get_wd14tag", {
        method: "POST",
        body: JSON.stringify(getStateAsJson()),
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

function showTagDialog() {
    const lora = document.getElementById("loraInput").value;
    const tags = lora.replace(/\n/g, ',').split(',').map(value => value.trim());
    const tagCenterPanel = document.getElementById("tagCenterPanel");

    for (const button of Array.from(tagCenterPanel.children)) {
        if (!button.classList.contains("disabled")) {
            button.classList.add("disabled");
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

    document.getElementById("tagDialog").style.display = "flex";
}

function tagSort() {
    const tagCenterPanel = document.getElementById("tagCenterPanel");
    Array.from(tagCenterPanel.children).sort((a, b) => a.value > b.value ? 1 : -1).forEach(node => tagCenterPanel.appendChild(node));
}

function tagRSort() {
    const tagCenterPanel = document.getElementById("tagCenterPanel");
    tagCenterPanel.children = Array.from(tagCenterPanel.children).sort((a, b) => Math.random() - 0.5).forEach(node => tagCenterPanel.appendChild(node));
}

function tagRSel() {
    const tagCenterPanel = document.getElementById("tagCenterPanel");
    Array.from(tagCenterPanel.children).forEach(child => Math.random() < 0.5 && child.click());
}

function tagCSel() {
    const tagCenterPanel = document.getElementById("tagCenterPanel");
    Array.from(tagCenterPanel.children).forEach(child => child.classList.contains("disabled") || child.classList.add("disabled"));
}

function tagOK() {
    const tagButtons = Array.from(tagCenterPanel.children);
    let selectedTags = [];
    for (let i = 0; i < tagButtons.length; i++) {
        if (!tagButtons[i].classList.contains("disabled")) {
            selectedTags.push(tagButtons[i].value);
        }
    }
    loraInput.value = selectedTags.join(", ");
    closeTagDialog();
}

function tagOK() {
    const tagButtons = Array.from(tagCenterPanel.children);
    let selectedTags = [];
    for (let i = 0; i < tagButtons.length; i++) {
        if (!tagButtons[i].classList.contains("disabled")) {
            selectedTags.push(tagButtons[i].value);
        }
    }
    loraInput.value = selectedTags.join(", ");
    closeTagDialog();
}

function closeTagDialog() {
    document.getElementById("tagDialog").style.display = "none";
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
        closeCaptureDialog();
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
    const offsetX = parseInt(document.getElementById("offsetXRange").value, 10);
    const offsetY = parseInt(document.getElementById("offsetYRange").value, 10);
    const scale = parseInt(document.getElementById("scaleRange").value, 10) / 100;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(scale, scale);
    ctx.translate(offsetX, offsetY);
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

function setFrame() {
    const canvas = document.getElementById("canvas");
    const dataURL = canvas.toDataURL("image/webp");
    fetch("/set_frame", {
        method: "POST",
        body: JSON.stringify([dataURL]),
        headers: { "Content-Type": "application/json" }
    }).then(response => {
        if (response.ok) {
            closeCaptureDialog();
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
    <div class="row">
        <div id="leftPanel">
            <textarea id="promptInput" placeholder="Enter prompt" rows="15">{param["prompt"]}</textarea>
            <textarea id="negativePromptInput" placeholder="Enter negativePrompt" rows="2">{param["negativePrompt"]}</textarea>
            <div class="row">
                Seed: <input id="seedInput" type="number" placeholder="Enter seed" value="{param["seed"]}" />
            </div>
            <div class="row">
                Keep: <input id="keepSeedInput" type="number" placeholder="keepSeed" value="{param["keepSeed"]}" />
            </div>
            <div class="row">
                Iv:<select id="intervalSelect" onchange="updateParam()">
                    <option value="10" {"selected" if param["interval"] == 10 else ""}>10</option>
                    <option value="15" {"selected" if param["interval"] == 15 else ""}>15</option>
                    <option value="20" {"selected" if param["interval"] == 20 else ""}>20</option>
                    <option value="25" {"selected" if param["interval"] == 25 else ""}>25</option>
                    <option value="30" {"selected" if param["interval"] == 30 else ""}>30</option>
                </select>
                <input id="autoUpdateCheckbox" type="checkbox" onchange="toggleAutoUpdate()"{" checked" if state["autoUpdate"] else ""}>Auto</input>
            </div>
            <div class="row">
                <button id="updateButton" class="willreload" onclick="updateParam(true)">U</button>
                <button id="changeSeedButton" onclick="changeSeed()">R</button>
                <button id="keepSeedButton" onclick="keepSeed()">K</button>
                <button id="backSeedButton" onclick="backSeed()">B</button>
            </div>
            <div class="row">
                <input id="stepsRange" type="range" min="1" max="50" step="1" value="{param["steps"]}" oninput="stepsValue.innerText = this.value;" />
                <span id="stepsValue">{param["steps"]}</span>stp
            </div>
            <div class="row">
                <input id="cfgRange" type="range" min="1.0" max="15.0" step="0.1" value="{param["cfg"]}" oninput="cfgValue.innerText = this.value;" />
                <span id="cfgValue">{param["cfg"]}</span>cfg
            </div>
            <div class="row">
                <input id="startstepRange" type="range" min="0" max="6" step="1" value="{param["startstep"]}" oninput="startstepValue.innerText = this.value;" />
                <span id="startstepValue">{param["startstep"]}</span>sta
            </div>
            <div class="row">
                <input id="framesRange" type="range" min="1" max="8" step="1" value="{param["frames"]}" oninput="framesValue.innerText = this.value;" />
                <span id="framesValue">{param["frames"]}</span>frm
            </div>
            <div class="row">
                <input id="framewaitRange" type="range" min="0" max="2" step="0.1" value="{param["framewait"]}" oninput="framewaitValue.innerText = this.value;" />
                <span id="framewaitValue">{param["framewait"]}</span>spf
            </div>
            <select id="samplerSelect" onchange="updateParam()">
                <option value="dpmpp_2m,sgm_uniform" {"selected" if param["sampler"] == "dpmpp_2m,sgm_uniform" else ""}>dpmpp_2m,sgm_uniform</option>
                <option value="ddim,ddim_uniform" {"selected" if param["sampler"] == "ddim,ddim_uniform" else ""}>ddim,ddim_uniform</option>
                <option value="eular,normal" {"selected" if param["sampler"] == "eular,normal" else ""}>eular,normal</option>
            </select>
            <select id="checkpointFolderSelect" class="willreload" onchange="updateParam(true)">
                <option value="" disabled selected>checkpoint folder</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["checkpointFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("ComfyUI/models/checkpoints", state["mode"]).glob("*/")])}
            </select>
            <div class="row">
                <select id="checkpointFileSelect">
                    <option value="" selected>checkpoint</option>
                    {f'<option value="{param["checkpoint"]}" selected>*{Path(param["checkpoint"]).stem}</option>' if param['checkpoint'] else ""}
                    {"".join([f'<option value="{file.relative_to("ComfyUI/models/checkpoints")}">{file.stem}</option>' if param["checkpoint"] != file.relative_to("ComfyUI/models/checkpoints") else "" for file in Path("ComfyUI/models/checkpoints", state["mode"], state["checkpointFolder"]).glob("*.safetensors")])}
                </select>
                <button id="moveCheckpointButton" onClick="moveCheckpoint()">M</button>
            </div>
            <select id="moveCheckpointSelect" onchange="moveCheckpointFile()">
                <option value="" disabled selected>move to</option>
                {"".join([f'<option value="{dir.name}">{dir.name}</option>' for dir in Path("ComfyUI/models/checkpoints", state["mode"]).glob("*/")])}
            </select>
        </div>
        <div id="centerPanel">
            <button id="toggleView" onclick="toggleView()"></button>
        </div>
        <div id="rightPanel">
            <select id="loraFolderSelect" class="willreload" onchange="updateParam(true)">
                <option value="" disabled selected>lora folder</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["loraFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("ComfyUI/models/loras", state["mode"]).glob("*/")])}
            </select>
            <div class="row">
                <select id="loraFileSelect" onchange="selectLoraFile()">
                    <option value="" disabled selected>lora file</option>
                    {"".join([f'<option value="{file.name}"{" selected" if state["loraFile"] == file.name else ""}>{file.stem}</option>' for file in Path("ComfyUI/models/loras", state["mode"], state["loraFolder"]).glob("*.safetensors")])}
                </select>
                <button id="toggleLoraButton" onclick="toggleLora()">T</button>
                <button id="moveLoraButton" onClick="moveLora()">M</button>
            </div>
            <select id="moveLoraSelect" onchange="moveLoraFile()">
                <option value="" disabled selected>move to</option>
                {"".join([f'<option value="{dir.name}">{dir.name}</option>' for dir in Path("ComfyUI/models/loras", state["mode"]).glob("*/")])}
            </select>
            <div class="row">
                <select id="loraTagSelect" onchange="toggleTag()">
                    <option value="" disabled selected>tags</option>
                    {"".join([f'<option value="{opt["value"]}"{" selected" if state["loraTag"] == opt["value"] else ""}>{opt["text"]}</option>' if opt["value"] else "" for opt in json.loads(state["loraTagOptions"])])}
                </select>
                <button id="toggleTagButton" onclick="toggleTag()">T</button>
                <button id="randomTagButton" onclick="randomTag()">R</button>
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
                <select id="videoSrcSelect" onchange="selectVideoSrc()">
                    <option value="" selected>video source</option>
                    {"".join([f'<option value="{file.name}"{" selected" if param["videosrc"] == file.name else ""}>{file.stem}</option>' for file in Path("videosrc").glob("*.*")])}
                </select>
                <button id="previewVideoButton" onclick="previewVideo()">PV</button>
            </div>
            <div class="row">
                <input id="videoFstRange" type="range" min="0" max="0" step="1" value="{param["videofst"]}" oninput="videoFstValue.innerText = this.value;" />
                <span id="videoFstValue">{param["videofst"]}</span>fst
            </div>
            <div class="row">
                <input id="videoSkpRange" type="range" min="1" max="32" step="1" value="{param["videoskp"]}" oninput="videoSkpValue.innerText = this.value;" />
                <span id="videoSkpValue">{param["videoskp"]}</span>skp
            </div>
            <div class="row">
                <input id="videoStrRange" type="range" min="0" max="2" step="0.01" value="{param["videostr"]}" oninput="videoStrValue.innerText = this.value;" />
                <span id="videoStrValue">{param["videostr"]}</span>str
            </div>
            <select id="presetFolderSelect" class="willreload" onchange="updateParam(true)">
                <option value="" disabled selected>preset folder</option>
                {"".join([f'<option value="{dir.name}"{" selected" if state["presetFolder"] == dir.name else ""}>{dir.name}</option>' for dir in Path("preset").glob("*/")])}
            </select>
            <select id="presetFileSelect">
                <option value="" disabled selected>preset</option>
                {"".join([f'<option value="{file.name}"{" selected" if state["presetFile"] == file.name else ""}>{file.stem}</option>' for file in Path("preset", state["presetFolder"]).glob("*.json")])}
            </select>
            <div class="row">
                <button id="randomPresetButton" onclick="randomPreset()">R</button>
                <button id="loadPresetLoraPromptButton" onclick="loadPreset(true)">P</button>
                <button id="loadPresetButton" class="willreload" onclick="loadPreset()">L</button>
                <button id="movePresetButton" onclick="movePreset()">M</button>
            </div>
            <select id="movePresetSelect" onchange="movePresetFile()">
                <option value="" disabled selected>move to</option>
                {"".join([f'<option value="{dir.name}">{dir.name}</option>' for dir in Path("preset").glob("*/")])}
            </select>
            <div class="row">
                <input id="presetTitleInput" placeholder="preset title" value="{state["presetTitle"]}" />
                <button id="savePresetButton" onclick="savePreset()">Save</button>
            </div>
            <div class="row">
                <button id="captureButton" onclick="capture()">C</button>
                <button id="wd14Button" onclick="addWD14Tag()">W</button>
                <button id="showTagDialogButton" onclick="showTagDialog()">T</button>
                <button id="updateButton2" onclick="updateParam()">U</button>
                <button id="clearLoraInputButton" onclick="clearLoraInput()">Clr</button>
            </div>
            <div class="row">
                <input id="wd14thRange" type="range" min="0" max="1" step="0.01" value="{state["wd14th"]}" oninput="wd14thValue.innerText = this.value;" />
                <span id="wd14thValue">{state["wd14th"]}</span>wth
            </div>
            <div class="row">
                <input id="wd14cthRange" type="range" min="0" max="1" step="0.01" value="{state["wd14cth"]}" oninput="wd14cthValue.innerText = this.value;" />
                <span id="wd14cthValue">{state["wd14cth"]}</span>cth
            </div>
            <textarea id="loraInput" placeholder="Enter lora" rows="15">{param["lora"]}</textarea>
            <div class="row">
                <a id="loraLink" href="{state["loraLinkHref"] or "javascript:void(0)"}" target="_blank">
                    <img id="loraPreview" src="{state["loraPreviewSrc"]}" alt onerror="this.onerror = null; this.src='';" />
                </a>
            </div>
        </div>
    </div>
    <div id="captureDialog">
        <div id="captureLeftPanel"></div>
        <div id="captureCenterPanel">
            <canvas id="canvas" width="960" height="600" />
        </div>
        <div id="captureRightPanel">
            <button id="resetPosButton" onclick="resetPos()">Reset</button>
            <br>
            X <input id="offsetXRange" type="range" min="-960" max="960" value="{param["offsetX"]}" onchange="updateCanvas()">
            <br>
            Y <input id="offsetYRange" type="range" min="-600" max="600" value="{param["offsetY"]}" onchange="updateCanvas()">
            <br>
            S <input id="scaleRange" type="range" min="50" max="400" value="{param["scale"]}" onchange="updateCanvas()">
            <br>
            <button id="setframeButton" onclick="setFrame()">SetFrame</button>
            <button id="closeCaptureButton" onclick="closeCaptureDialog()">Close</button>
        </div>
    </div>
    <div id="tagDialog">
        <div id="tagLeftPanel"></div>
        <div id="tagCenterPanel"></div>
        <div id="tagRightPanel">
            <button id="tagSortButton" onclick="tagSort()">Sort</button>
            <button id="tagRSortButton" onclick="tagRSort()">RSort</button>
            <br>
            <button id="tagRSelButton" onclick="tagRSel()">RSel</button>
            <button id="tagCSelButton" onclick="tagCSel()">CSel</button>
            <br>
            <button id="tagOKButton" onclick="tagOK()">OK</button>
            <button id="tagCancelButton" onclick="closeTagDialog()">Cancel</button>
        </div>
    </div>
    <script>
    {SCRIPT_PARAM}
    {SCRIPT_PRESET}
    {SCRIPT_CKPT}
    {SCRIPT_LORA}
    {SCRIPT_VID}
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
    loraPath = Path("ComfyUI/models/loras", state["mode"], req["loraFolder"], req["loraFile"])
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


@server.PromptServer.instance.routes.post("/get_wd14tag")
async def get_wd14tag(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()
    
    stt = await request.json()
    state.update(stt);
    tags = []
    if frame_buffer:
        with BytesIO(frame_buffer[0]) as buf:
            image = Image.open(buf)
            tags = await wd14tagger.tag(image, "wd-v1-4-moat-tagger-v2.onnx", state["wd14th"], state["wd14cth"], exclude_tags)
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


def load_video(path, height, videofst, videoskp, maxcount):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, videofst)
    buf = []
    for i in range(maxcount):
        for s in range(videoskp + 1):
            if not cap.isOpened():
                return buf, total_frames
            if not cap.grab():
                return buf, total_frames
        _, frame = cap.retrieve()
        if height != frame.shape[0]:
            scale = height / frame.shape[0]
            width = int(frame.shape[1] * scale // 8 * 8)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        buf.append(frame)
    return buf, total_frames


@server.PromptServer.instance.routes.post("/preview_video")
async def preview_video(request):
    global frame_buffer
    if request.remote not in allowed_ips:
        raise HTTPForbidden()
    
    videosrc, videofst, videoskp, frames = await request.json()
    buf, total_frames = load_video(str(Path("videosrc", videosrc)), state["height"], videofst, videoskp, frames)
    buf = [BytesIO(cv2.imencode(".webp", frame)[1]).getvalue() for frame in buf]
    buf += buf[::-1]
    frame_buffer = buf
    return web.json_response({"total_frames": total_frames})


@server.PromptServer.instance.routes.post("/move_checkpointfile")
async def move_lorafile(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, moveFrom, moveTo = await request.json()
    state.update(stt)
    pathFrom = Path("ComfyUI/models/checkpoints", state["mode"], state["checkpointFolder"], Path(moveFrom).name)
    pathTo = Path("ComfyUI/models/checkpoints", state["mode"], moveTo, pathFrom.name)
    Path(pathFrom).rename(pathTo)
    state["checkpointFolder"] = moveTo
    return web.Response(status=200)


@server.PromptServer.instance.routes.post("/move_presetfile")
async def move_presetfile(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, moveTo = await request.json()
    state.update(stt)
    pathFrom = Path("preset", state["presetFolder"], state["presetFile"])
    pathTo = Path("preset", moveTo, pathFrom.name)
    Path(pathFrom).rename(pathTo)
    state["presetFolder"] = moveTo
    return web.Response(status=200)


@server.PromptServer.instance.routes.post("/move_lorafile")
async def move_lorafile(request):
    if request.remote not in allowed_ips:
        raise HTTPForbidden()

    stt, moveTo = await request.json()
    state.update(stt)
    pathFrom = Path("ComfyUI/models/loras", state["mode"], state["loraFolder"], state["loraFile"])
    pathTo = Path("ComfyUI/models/loras", state["mode"], moveTo, pathFrom.name)
    Path(pathFrom).rename(pathTo)
    state["loraFolder"] = moveTo
    return web.Response(status=200)


@server.PromptServer.instance.routes.post("/load_preset")
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
                data["lora"] = base64.b64decode(buf["lora"]).decode("utf-8")
        else:
            data = buf
            data["prompt"] = base64.b64decode(buf["prompt"]).decode("utf-8")
            data["negativePrompt"] = base64.b64decode(buf["negativePrompt"]).decode("utf-8") if "negativePrompt" in buf else ""
            data["lora"] = base64.b64decode(buf["lora"]).decode("utf-8")

    param.update(data)
    state["presetTitle"] = Path(state["presetFile"]).stem
    return web.json_response({"lora": data["lora"], "presetTitle": state["presetTitle"]})


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


class FlipStreamLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "default_ckpt": (checkpoints_list,),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "loader"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, default_ckpt):
        return hash(param["checkpoint"])
    
    def loader(self, default_ckpt):
        global frame_updating
        frame_updating = True
        ckpt_path = None
        if param["checkpoint"]:
            ckpt_path = folder_paths.get_full_path("checkpoints", param["checkpoint"])
        if ckpt_path is None:
            ckpt_path = folder_paths.get_full_path("checkpoints", default_ckpt)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]


class FlipStreamSource:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
            },
        }

    RETURN_TYPES = ("IMAGE","LATENT","BOOLEAN","INT","INT","INT","FLOAT")
    RETURN_NAMES = ("image","latent","bypass","width","height","frames","videostr")
    OUTPUT_IS_LIST = (True,True,False,False,False,False,False)

    FUNCTION = "source"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, vae, width, height):
        state["height"] = height
        return hash((width, height, param["frames"], param["videosrc"], param["videofst"], param["videoskp"], param["videostr"]))

    def source(self, vae, width, height):
        videopath = str(Path("videosrc", param["videosrc"])) if param["videosrc"] else ""
        frames = param["frames"]
        image = None
        latent = None
        bypass = False
        if videopath:
            buf, _ = load_video(videopath, height, param["videofst"], param["videoskp"], frames)
            if buf:
                buf = [np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.float32) / 255 for frame in buf]
                image = torch.zeros([frames, height, width, 3])
                image2 = torch.from_numpy(np.fromiter(buf, np.dtype((np.float32, buf[0].shape))))
                x2 = image.shape[2] // 2
                w2 = image2.shape[2] // 2
                image[:, :, x2 - w2: x2 + w2] = image2
                latent = vae.encode(image)
                bypass = True
        if image is None:
            image = torch.zeros([frames, height, width, 3])
            latent = torch.zeros([frames, 4, height // 8, width // 8], device=self.device)
        frames, h, w, _ = image.shape
        images = [image[i:i + 1, ...] for i in range(image.shape[0])]
        latents = [{"samples": latent[i:i + 1, ...]} for i in range(latent.shape[0])]
        return (images, latents, bypass, w, h, frames, param["videostr"])


class FlipStreamUpdate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("prompt","batchPrompt","appPrompt","negativePrompt")
    FUNCTION = "update"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls):
        return hash((param["prompt"], param["negativePrompt"], param["lora"]))
        
    def update(self):
        global frame_updating
        frame_updating = True
        buf = param["prompt"].split("----")
        prompt = buf[0].replace("{lora}", param["lora"]).strip()
        batchPrompt = buf[1].strip() if len(buf) > 1 else "-\n-\n"
        appPrompt = buf[2].replace("{lora}", param["lora"]).strip() if len(buf) > 2 else ""
        batchPrompt = ",\n".join([f'"{n}":"{item.lstrip("-").strip()}"' for n, item in enumerate(batchPrompt.split("\n"))])
        return (prompt, batchPrompt, appPrompt, param["negativePrompt"])


class FlipStreamPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "batchPrompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "appPrompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "frames": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("framePrompt",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "encode"
    CATEGORY = "FlipStreamViewer"

    def encode(self, prompt, batchPrompt, appPrompt, frames):
        framePrompt = []
        item = prompt
        s = json.loads("{" + batchPrompt + "}")
        for i in range(frames):
            if i < len(s):
                item = " ".join([prompt, s[str(i)], appPrompt]).strip()
            framePrompt.append(item)
        return (framePrompt,)


class FlipStreamOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("INT","INT","FLOAT","INT",comfy.samplers.KSampler.SAMPLERS,comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("seed","steps","cfg","startstep","sampler","scheduler")
    FUNCTION = "option"
    CATEGORY = "FlipStreamViewer"

    @classmethod
    def IS_CHANGED(cls, mode):
        state["mode"] = mode
        return hash((mode, param["seed"], param["steps"], param["cfg"], param["startstep"], param["sampler"]))
    
    def option(self, mode):
        global frame_updating
        frame_updating = True
        sampler, scheduler = param["sampler"].split(",")
        return (param["seed"], param["steps"], param["cfg"], param["startstep"], sampler, scheduler)


class FlipStreamSwitchVFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("IMAGE",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE","BOOLEAN")
    RETURN_NAMES = ("tensor","bypass")
    FUNCTION = "control"
    CATEGORY = "FlipStreamViewer"

    def control(self, tensor):
        tensor = torch.cat(tensor, dim=0)
        return (tensor, tensor.shape[0] >= 2)


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
    def IS_CHANGED(cls, tensor, allowip, wd14exc, idle):
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

    def update_frame(self, tensor, allowip, wd14exc, idle):
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
    "FlipStreamSource": FlipStreamSource,
    "FlipStreamUpdate": FlipStreamUpdate,
    "FlipStreamPrompt": FlipStreamPrompt,
    "FlipStreamOption": FlipStreamOption,
    "FlipStreamSwitchVFI": FlipStreamSwitchVFI,
    "FlipStreamViewer": FlipStreamViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlipStreamLoader": "FlipStreamLoader",
    "FlipStreamSource": "FlipStreamSource",
    "FlipStreamUpdate": "FlipStreamUpdate",
    "FlipStreamPrompt": "FlipStreamPrompt",
    "FlipStreamOption": "FlipStreamOption",
    "FlipStreamSwitchVFI": "FlipStreamSwitchVFI",
    "FlipStreamViewer": "FlipStreamViewer",
}
