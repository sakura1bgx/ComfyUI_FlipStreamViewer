# ComfyUI_FlipStreamViewer

**ComfyUI_FlipStreamViewer** is a tool that provides a viewer interface for flipping images with frame interpolation, allowing you to watch high-fidelity pseudo-videos without needing AnimateDiff.

![sample](https://github.com/user-attachments/assets/4ceecd68-bd35-4eb5-95bf-4ce822ff0591)

## Required Custom Nodes

Please install the following nodes via ComfyUI-Manager:

- ComfyUI Impact Pack
- ComfyUI Frame Interpolation
- ComfyUI WD 1.4 Tagger
- ComfyUI-Advanced-ControlNet
- ComfyUI-AutomaticCFG
- LoRA Tag Loader for ComfyUI

## Getting Started
1. **Run ComfyUI** and from ComfyUI-Manager, select Manager -> Custom Node Manager and install ComfyUI_FlipStreamViewer.
2. **Restart ComfyUI** and load the `workflow.json` file located in the `workflows` folder.
3. Access the viewer at the following URL if ComfyUI is running at `http://localhost:8188`:

`http://localhost:8188/viewer`

4. To respond to viewer input:
- Check 'Extra options', 'Auto Queue', and 'instant' on the ComfyUI control panel.
- Then click the 'Queue Prompt' button.

5. The 'preset' folder may be created in the current directory for saving and loading preset files. Sample presets can be found in the `presetsamples` folder.

## Prompt Format

Use the following format for input prompts. The 'pre text' and 'append text' sections apply to all frames. The separator `----` should consist of four hyphens, and each line of 'frameN text' applies to individual frames. The '{lora}' means it will be replaced by the LoRA input contents at the right panel.

```
pre text,
{lora}
----
- frame1 text
- frame2 text
----
append text
```

### Limitations

- Please note that this tool is not designed with security measures against third-party attacks. Therefore, it should only be used for personal purposes within a secure network.
- There is no save video function available; please customize your workflow as needed.
- Excessively long text may cause exceptions.
- The keywords `BREAK`, `AND` cannot be used in the prompt.

## Parameters

You can adjust the following parameters on the viewer:

- **Seed**: Random seed for KSampler.
- **Keep**: Keeped seed.
- **Iv**: Interval in seconds for auto-changing the random seed.
- **Nstp**: Steps for KSampler.
- **Ncfg**: CFG for KSampler.
- **Nsta**: Start step for the second KSampler; increasing this value will stabilize the output between frames.
- **Nfrm**: Number of frames; each frame undergoes x8 interpolation in the sample workflow.
- **Nspf**: Seconds per frame displayed in the viewer stream.
- **Rate**: LoRA late for toggle and add tags rank.
- **Nfst**: Seek first frame for video source
- **Nskp**: Skip frames for video source
- **Nstr**: Strength for video source controlnet
- **Nwth**: Threshould for WD14Tagger.
- **Ncth**: Character threshould for WD14Tagger.

## Inputs

Provide the following text inputs in the viewer:

- **Prompt Input**: Located at the top of the left panel (refer to the Prompt Format section above).
- **Negative Prompt Input**: Located below the Prompt Input; this applies to all frames.
- **LoRA Input**: Located in the middle of the right panel; input LoRA such as `<lora:startsfilename:1>`.

## Controls

**Left Panel Buttons:**

- **U**: Apply input data to the workflow.
- **R**: Change the random seed and update.
- **K**: Keep the seed to search for another good seed.
- **B**: Go back to the previous seed.
- **M**: Move the checkpoint file.

**Right Panel Buttons:**

- **T**: Toggle LoRA enable/disable.
- **M**: Move the LoRA file.
- **T**: Toggle tag enable/disable at the LoRA Input.
- **R**: Add random tags at the LoRA Input.
- **PV**: Preview video source.
- **R**: Select random preset.
- **P**: Load a preset prompt only.
- **L**: Load a preset.
- **M**: Move the Preset file.
- **Save**: Save the current preset.
- **C**: Capture an image from another browser tab or desktop and set it as the first frame; useful with W.
- **W**: Apply WD14Tagger to the first frame; outputs tags to LoRA input.
- **T**: Toggle tags in LoRA input. 
- **U**: Update without reload; useful with T.
- **Clr**: Clear the LoRA input.

## Additional Features

- **LoRA Preview**: At the bottom of the right panel; click to jump to the Civitai LoRA page if found.
- **Auto-hide**: Click to hide controls and images or the viewer auto-hides after 5 minutes of inactivity (reload).

## For More Quality

- Use more higher-quality checkpoint models.
- To enhance stability between frames, use a deliberate prompt in the 'pre text' to establish a consistent background, such as `ideal background, on desk`.
- Select the appropriate words or LoRA in 'frameN text' to achieve the desired motion.

## For More Speed

- ComfyUI commandline options '--highvram --fast --fp8_e4m3fn-unet' are good for speed, however it depends on system and model.

## Customizing Workflow

You can customize the workflow for specific needs:

- **FlipStreamLoader**: Use the 'mode' parameter to switch between sub-directories for checkpoints and LoRAs, useful for managing models like sd15, pony, etc.
- **FlipStreamSetMode**: Only change the 'mode' parameter for the workflow without checkpoints.
- **FlipStreamViewer**: The 'allowip' parameter allows you to set IP addresses that can access the viewer. The 'w14exc' parameter is used to set exclude_tags for the WD14 Tagger.
