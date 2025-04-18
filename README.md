# ComfyUI_FlipStreamViewer

**ComfyUI_FlipStreamViewer** is a tool that provides a customizable viewer interface for flipping images with frame interpolation.

<div><video controls src="https://github.com/user-attachments/assets/2a40c7c6-045e-4d86-a0fd-51a24f5472b4"></video></div>

## Getting Started
1. **Run ComfyUI** and from ComfyUI-Manager, select Manager -> Custom Node Manager and install ComfyUI_FlipStreamViewer.
2. **Restart ComfyUI** and load the `simple.json` file located in the `workflows` folder.
3. Access the viewer at the following URL if ComfyUI is running at localhost port 8188:

`http://localhost:8188/flipstreamviewer`

4. To respond to viewer input:
- Check 'Extra options', 'Auto Queue', and 'instant' on the ComfyUI control panel.
- Then click the 'Queue Prompt' button.

5. The UI nodes will refresh after the workflow is completed once, so you need to click 'Update and reload' after that.

6. The 'preset' folder may be created in the current directory for saving and loading preset files.

## Limitations

- Please note that this tool is not designed with security measures against third-party attacks. Therefore, it should only be used for personal purposes within a secure network.
- There is no save video function available; please customize your workflow as needed.

## Optional Custom Nodes dependencies

The following custom nodes can be used as needed:
- ComfyUI-WD14-Tagger: For Tagger
- ComfyUI-Inspyrenet-Rembg: For FlipStreamRembg
- ComfyUI-Florence2: For FlipStreamSegMask
- ComfyUI-Frame-Interpolation: For FlipStreamFilmVfi

The following custom nodes may also be used within workflows:
- ComfyUI Impact Pack
- LoRA Tag Loader for ComfyUI
- ComfyUI-AnimateDiff-Evolved
- ComfyUI-VideoHelperSuite
- ComfyUI-DepthAnythingV2
- ComfyUI-Advanced-ControlNet

## Optional Python Packages Dependencies

llama-cpp-python for FlipStreamLoadChatModel:
```
cd python_embeded
python -m pip install scikit-build-core
python -m pip install cmake
set CMAKE_ARGS="-DGGML_CUDA=on"
python -m pip install llama-cpp-python
```

## UI Features

- **Customizable Left Panel**: The left panel UI is customizable with UI Nodes.
- **UI Node Ordering**: These UI nodes are ordered by their titles, and you can change the node titles to something like '10.steps' or '11.cfg'. You can also click on the title to compact the node on the workflow.
- **Label Input**: Each UI node has a 'label' input, which should be a unique identifier used as the parameter name. 'lora' is internally used, so they cannot be used for labels.
- **Enable Output**: Each UI node also has an 'enable' output to use with 'Control Bridge' for switching to bypass some nodes or FlipStreamSwitch* for switching input.
- **Right Panel**: You can operate several prepared functions such as Status, Darker, Tagger, Preset, and Lora.
- **Status**: The right panel shows ComfyUI status and error information.
- **Darker**: You can adjust the image brightness on the viewer using the darker parameter. It can also be set via query:
`http://localhost:8188/flipstreamviewer?darker=0.33`
- **Tagger**: You can capture screenshots and generate tags from images using WD14. It depends on ComfyUI-WD14-Tagger.
- **Preset**: You can save and load parameters that are set on viewer controls. The 'M' button means Move to another folder. It can also be set via query:
`http://localhost:8188/flipstreamviewer?showPresetDialog=1&presetFolder=folder_name&presetFile=file_name.json`
- **Lora**: You can select LoRA and choose tags. It can also choose random tags. The LoRA preview box can be clicked to jump to the Civitai LoRA page if found. The 'M' button means Move to another folder. The 'T' button means Toggle. The 'R' button means Random choose.
- **Toggle View**: You can click the center panel to hide the left and right panels, and a subsequent click will hide the image. It can also be set via query: 
`http://localhost:8188/flipstreamviewer?toggleView=1`
- **Message Box**: If a message is set by FlipStreamSetMessage, you can view it at the bottom of the center panel.
- **Auto Hide**: Stream and preview in the viewer will automatically hide after 5 minutes if the page is not reloaded.

## UI Nodes

- **FlipStreamSection**: A section label for UI Nodes.
- **FlipStreamSlider**: A slider for adjusting values.
- **FlipStreamTextBox**: A text box for inputting multiline text.
- **FlipStreamInputBox**: An input box for various boxtype inputs. The 'U' button means update. You can choose special boxtype 'seed' or 'r4d' with the 'R' button, which means randomize. The boxtype 'r4d' can be used to generate a 4-digit part of a prompt like 'MOV_{num}'. In this case, FlipStreamTextReplace can help find '{num}' and replace with the output of the input box.
- **FlipStreamSelectBox_Samplers**: A select box for choosing samplers.
- **FlipStreamSelectBox_Scheduler**: A select box for choosing schedulers.
- **FlipStreamFileSelect_Checkpoints**: A file selector for checkpoints. 'mode' is used to select a subfolder such as 'sd15', 'sdxl', 'pony', or 'flux' in 'checkpoints'. 'use_sub' means using subfolders in the 'mode' folder. 'use_move' means using the move file selector.
- **FlipStreamFileSelect_VAE**: A file selector for VAE models.
- **FlipStreamFileSelect_ControlNetModel**: A file selector for ControlNet models.
- **FlipStreamFileSelect_TensorRT**: A file selector for TensorRT models. It may be used with ComfyUI_TensorRT.
- **FlipStreamFileSelect_AnimateDiffModel**: A file selector for AnimateDiff models. It may be used with ComfyUI-AnimateDiff-Evolved.
- **FlipStreamFileSelect_Input**: A file selector for ComfyUI input folder.
- **FlipStreamFileSelect_Output**: A file selector for ComfyUI output folder.
- **FlipStreamPreviewBox**: A box for previewing the input image. You can select the ROI (Region of Interest) in the preview by dragging the mouse.

## Other Nodes

- **FlipStreamSetUpdateAndReload**: Updates parameters and reloads pages after a delay.
- **FlipStreamSetMessage**: Sets a message for the message box at the bottom of the center panel.
- **FlipStreamSetParam**: A node for setting parameters. However, it needs a reload to update the value in the viewer UI.
- **FlipStreamGetParam**: A node for getting parameters. 'lora' is a prepared parameter that contains text at the Lora input. Some prompts can contain '{lora}' and FlipStreamTextReplace can help find '{lora}' and replace it with the output of the get param node of 'lora'. 'b64dec' is true for 'lora' and other 'FlipStreamTextBox' parameters, because these multiline parameters are internally base64 encoded.
- **FlipStreamGetPreviewRoi**: A node for obtaining the preview ROI (Region of Interest) selection.
- **FlipStreamImageSize**: A node for getting image size.
- **FlipStreamTextReplace**: A node for replacing text. It will output the result of `text.replace(find, replace.format(value))`.
- **FlipStreamScreenGrabber**: A node for grab multiframe screenshots.
- **FlipStreamSource**: A node for prepare image or latent source.
- **FlipStreamSwitchImage**: A node for switching images.
- **FlipStreamSwitchLatent**: A node for switching latents.
- **FlipStreamGate**: A node for consolidating input timing to ensure that inputs to the sampler do not occur sequentially.
- **FlipStreamRembg**: A node for remove background. It depends on ComfyUI-Inspyrenet-Rembg.
- **FlipStreamSegMask**: A node for segmentation masks. The target can contain multiple words separated by commas for segmentation. It will use the 'microsoft/Florence-2-large' model, which you can download using the DownloadAndLoadFlorence2Model node of ComfyUI-Florence2. Segmentation sometimes fails, so you may need to try some other random seeds.
- **FlipStreamChat**: Loads an LLM (Large Language Model) and obtains chat responses. It only supports *.gguf files in the models/LLM folder and uses llama-cpp-python.
- **FlipStreamBatchPrompt**: A node for simple batch prompting. Use the following format for the input prompt of this node. The 'pre text' and 'append text' sections apply to all frames. The separator `----` should consist of four hyphens, and each line of 'frame text' applies evenly to the number of frames specified in 'frames'. For example, if 'frames' is 8 and there are 2 'frame text' lines, they will be applied starting from frames 0 and 4.
```
pre text,
----
- frame text,
- frame text,
----
append text
```
- **FlipStreamFilmVfi**: A node for video frame interpolation. It depends on ComfyUI-Frame-Interpolation.
- **FlipStreamViewer**: A node for viewing content. The 'allowip' parameter allows you to set IP addresses that can access the viewer. ComfyUI commandline options "--listen 0.0.0.0" and appropriate firewall settings are also needed for that. The 'w14exc' parameter is used to set exclude_tags for the WD14 Tagger. 'fps' can control flip speed.

## For More Quality

- Use more higher-quality checkpoint models.
- To enhance stability between frames, use a deliberate prompt in the 'pre text' to establish a consistent background, such as `ideal background, on desk`.
- Select the appropriate words or LoRA in 'frame text' to achieve the desired motion.

## For More Speed

- ComfyUI commandline options '--fast --fp8_e4m3fn-unet' are good for speed, however it depends on system and model.

## Workflow Examples

- **simple.json**: Customized a ComfyUI default workflow for FlipStreamViewer UI nodes.
- **dmd2_lora.json**: Using DMD2 and some LoRA, only needs 5 steps to generate an image.
- **segmask.json**: A 2-step process for generating frames using batch prompts and segmentation mask, followed by interpolation.
- **animate.json**: Using AnimateDiff with FlipStreamSource, FlipStreamBatchPrompt, and FlipStreamFilmVfi.
- **grab_rembg_depth_animate.json**: Using ControlNet and AnimateDiff with FlipStreamScreenGrabber, FlipStreamRembg.
- **quick_vid2vid.json**: Quick tuning workflow for vid2vid.
- **quick_vid2vid_roi.json**: An example demonstrating the use of FlipStreamGetPreviewRoi and FlipStreamGate.
- **visualnobel.json**: Visual nobel like UI using FlipStreamChat.
