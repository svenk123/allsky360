# Timelapse Creator with CUDA Acceleration (nvJPEG + libx264)

This project is a high-performance C/CUDA-based program that converts a series of JPEG images into a timelapse video (MP4 format). JPEG decoding and resizing run on the GPU through NVIDIA libraries, the H.264 encoding runs on the CPU:

- **nvJPEG** for GPU-accelerated JPEG decoding
- **NPP** (NVIDIA Performance Primitives) for image resizing on the GPU
- **libx264** (via FFmpeg) for H.264 encoding on the CPU. The Jetson Orin Nano has no hardware video encoder (NVENC)
- Optional **progress bar** for real-time feedback

## Features

- Input: List of `.jpg` images
- Output: MP4 video using H.264 compression (libx264, CRF 23, peak rate ~4 Mbps)
- Configurable output resolution via `-w` and `-h`
- Configurable frame rate via `-p`
- GPU-accelerated JPEG decoding and scaling
- Optional progress display via `-v`

## Requirements

- Linux system with NVIDIA GPU
- CUDA Toolkit (e.g. 11.x or 12.x)
- FFmpeg with development headers (`libavcodec`, `libavformat`, `libavutil`, `libswscale`)
- NVIDIA libraries: `nvJPEG`, `NPP`
- FFmpeg built with `libx264`
- Compiler: `gcc` and `nvcc`

## Build Instructions

```bash
make

This will create the binary bin/allsky360-timelapse.

## Example Usage
```
./bin/allsky360-timelapse -o output.mp4 -w 1920 -h 1080 -p 25 -v img001.jpg img002.jpg img003.jpg ...

### Parameters:
- -o output.mp4: Output filename
- -w 1920: Target video width
- -h 1080: Target video height
- -p 25: Frames per second
- -v: (Optional) Enable progress bar
- img001.jpg ...: List of input images (in desired order)

## Performance Benefits
	•	Faster JPEG Decoding: nvJPEG uses GPU memory and pipelines to decode images much faster than traditional libjpeg.
	•	Zero-Copy Resizing: Images are resized directly on the GPU using NPP, avoiding costly memory transfers.
	•	Scalable: Can process hundreds or thousands of frames efficiently on CUDA-capable GPUs.


## Mixed image sizes

A config change during the night can change the image size. All JPEG headers are read first:

- **Normal timelapse:** every image is used. Each one is scaled with its aspect ratio kept and centred in the video frame, black bars fill the rest (letterbox).
- **Stacking modes** (`-m avg|max|min|sigma|diff|motion`): pixels must line up, so only the size with the most images is used (tie: the size of the newest image). The other images are skipped with one summary warning.
- Unreadable or broken JPEGs are skipped instead of aborting the run.

## Notes
	•	For optimal performance, use images of similar dimensions and consistent formatting.
	•	Encoder settings (CRF 23, peak rate ~4 Mbps, preset slow) are set in video_encoder.c.
	•	Designed for use in scientific imaging, astronomy, and observatory camera projects (e.g., allsky).

## License

MIT License
