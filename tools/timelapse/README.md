# Timelapse Creator with CUDA Acceleration (nvJPEG + NVENC)

This project is a high-performance C/CUDA-based program that converts a series of JPEG images into a timelapse video (MP4 format). The implementation uses GPU acceleration through NVIDIA technologies for both JPEG decoding and video encoding:

- **nvJPEG** for GPU-accelerated JPEG decoding
- **NPP** (NVIDIA Performance Primitives) for image resizing on the GPU
- **NVENC** for fast H.264 video encoding on the GPU
- Optional **progress bar** for real-time feedback

## Features

- Input: List of `.jpg` images
- Output: MP4 video using H.264 compression (via NVENC)
- Configurable output resolution via `-w` and `-h`
- Configurable frame rate via `-p`
- GPU-accelerated JPEG decoding and scaling
- GPU-accelerated video encoding
- Optional progress display via `-v`

## Requirements

- Linux system with NVIDIA GPU
- CUDA Toolkit (e.g. 11.x or 12.x)
- FFmpeg with development headers (`libavcodec`, `libavformat`, `libavutil`, `libswscale`)
- NVIDIA libraries: `nvJPEG`, `NPP`, `NVENC`
- Compiler: `gcc` and `nvcc`

## Build Instructions

```bash
make

This will create the binary timelapse_cuda_nvenc.

## Example Usage
```
./timelapse_cuda_nvenc -o output.mp4 -w 1920 -h 1080 -p 25 -v img001.jpg img002.jpg img003.jpg ...

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
	•	Real-time Encoding: NVENC allows high-quality H.264 encoding with low CPU load and high throughput.
	•	Scalable: Can process hundreds or thousands of frames efficiently on CUDA-capable GPUs.


## Notes
	•	For optimal performance, use images of similar dimensions and consistent formatting.
	•	Output bitrate is currently fixed to ~4 Mbps; this can be modified in video_encoder.c.
	•	Designed for use in scientific imaging, astronomy, and observatory camera projects (e.g., allsky).

## License

MIT License
