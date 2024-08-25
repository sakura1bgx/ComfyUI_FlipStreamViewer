# ComfyUI_FlipStreamViewer

**ComfyUI_FlipStreamViewer** is a tool that provides a viewer interface for flipping images with frame interpolation, allowing you to watch high-fidelity pseudo-videos without needing AnimateDiff.

![viewer_snapshot](https://github.com/user-attachments/assets/61e79e55-111e-40b9-8417-f93acb90aed7)

## Required Custom Nodes

Please install the following nodes via ComfyUI-Manager:

- ComfyUI Impact Pack
- ComfyUI Frame Interpolation
- ComfyUI WD 1.4 Tagger
- FizzNodes
- LoRA Tag Loader for ComfyUI

## Getting Started
1. Navigate to the custom_nodes directory of ComfyUI. Run the following command to clone the repository:

`git clone https://github.com/sakura1bgx/ComfyUI_FlipStreamViewer.git`

2. **Run ComfyUI** and load the `workflow.json` file located in the `workflows` folder.
3. Access the viewer at the following URL if ComfyUI is running at `http://localhost:8188`:

`http://localhost:8188/viewer`

3. To respond to viewer input:
- Check 'Extra options', 'Auto Queue', and 'instant' on the ComfyUI control panel.
- Then click the 'Queue Prompt' button.

4. The 'preset' folder may be created in the current directory for saving and loading preset files. Sample presets can be found in the `presetsamples` folder.

## Prompt Format

Use the following format for input prompts. The 'pre text' and 'append text' sections apply to all frames. The separator `----` should consist of four hyphens, and each line of 'frameN text' applies to individual frames.

```
pre text,
----
- frame1 text
- frame2 text
----
append text
```

### Limitations

- There is no save video function available; please customize your workflow as needed.
- Excessively long text may cause exceptions.
- The keywords `BREAK`, `AND`, and `<lora:...>` cannot be used in the prompt.

## Parameters

You can adjust the following parameters on the viewer:

- **Seed**: Random seed for KSampler.
- **Iv**: Interval in seconds for auto-changing the random seed.
- **Nstp**: Steps for KSampler.
- **Ncfg**: CFG for KSampler.
- **Nsta**: Start step for the second KSampler; increasing this value will stabilize the output between frames.
- **Nfrm**: Number of frames; each frame undergoes x8 interpolation in the sample workflow.
- **Nspf**: Seconds per frame displayed in the viewer stream.

## Inputs

Provide the following text inputs in the viewer:

- **Prompt Input**: Located at the top of the left pane (refer to the Prompt Format section above).
- **Negative Prompt Input**: Located below the Prompt Input; this applies to all frames.
- **LoRA Input**: Located in the middle of the right pane; input LoRA such as `<lora:startsfilename:1>`.

## Controls

**Left Pane Buttons:**

- **Update**: Apply input data to the workflow.
- **Change**: Change the seed and update.
- **K**: Keep the seed to search for another good seed.
- **B**: Go back to the previous seed.
- **M**: Move the checkpoint file.
- **L**: Load a preset.
- **P**: Load a preset prompt only.
- **S**: Save the current preset.
- **D**: Delete a preset.

**Right Pane Buttons:**

- **T**: Toggle LoRA enable/disable.
- **M**: Move the LoRA file.
- **T**: Toggle tag enable/disable at the Prompt Input.
- **R**: Switch to random tags at the Prompt Input.
- **U**: Same as Update.
- **W**: Apply WD14Tagger to the first frame; outputs tags to LoRA input.
- **C**: Capture an image from another browser tab or desktop and set it as the first frame; useful with W.

## Additional Features

- **LoRA Preview**: At the bottom of the right pane; click to jump to the Civitai LoRA page if found.
- **Auto-hide**: Click to hide controls and images or the viewer auto-hides after 5 minutes of inactivity.

## Quality and Speed

- Use more higher-quality checkpoint models.
- To enhance stability between frames, use a deliberate prompt in the 'pre text' to establish a consistent background, such as `on desk, ideal background`. Additionally, using LORA can also improve stability.
- Select the appropriate words in 'frameN text' to achieve the desired motion. This can be reused in other prompts.
- Since 0sta is more flexible, it's better to change the Nsta only after testing those first.
- Recommended speed setting: `4stp 1cfg 0sta dpmpp_2m,sgm_uniform` with `<lora:TCD-SDXL:1>`. You can download the TCD LoRA from [Civitai](https://civitai.com/models/395638). Rename the file to `TCD-SDXL.safetensors`.

## Customizing Workflow

You can customize the workflow for specific needs:

- **FlipStreamLoader**: Use the 'mode' parameter to switch between sub-directories for checkpoints and LoRAs, useful for managing models like sd15, pony, etc.
