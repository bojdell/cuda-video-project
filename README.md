CUDA Video Processing Project
========================

This project utilizes parallel computing to improve the efficiency of post-processing  video footage.

###How To Process A Video:

1. Find a video you want to process, and put it in `./input_videos`

2. Download and install [FFmpeg](https://ffmpeg.org/)

3. Use FFmpeg to break the video into frames. This code breaks off the first 20 frames of video `sample_video`:

  ```
  ffmpeg -i input_videos/sample_video.mp4 -vframes 20 infiles/tmp%03d.ppm
  ```

4. Compile the project code

 - For serial implementation (slower execution): `make serial`
 - For parallel implementation (faster execution, requires [CUDA](https://developer.nvidia.com/cuda-downloads) and a valid GPU): `make parallel`

  *Note: `make` defaults to `make parallel`*

5. Run the code: 
 - Serial: `./ppm_serial <stride_len>`
 - Parallel: `./ppm <stride_len>`

  Where `stride_len` is an optional parameter that can be used to control the number of frames stored in memory at one time. The default value is 20.

6. Use FFmpeg to combine the output frames into a video:
```
ffmpeg -framerate 24 -i outfiles/tmp%03d.ppm -c:v libx264 -r 30 -pix_fmt yuv420p output_video.mp4
```