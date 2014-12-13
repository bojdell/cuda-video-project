#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <time.h>
#include "ppmKernel.cu"
#include "ppm.h"

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

static PPMImage *readPPM(const char *filename)
{
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

// Mallocs space for a filter, initializes it, and returns a pointer to it
Filter3D * initializeFilter()
{
    double data[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE] =  { { {0, 0, 0, 0, 0},
                                                               {0, 0, 0, 0, 0},
                                                               {0, 0, 0, 0, 0},
                                                               {0, 0, 0, 0, 0},
                                                               {0, 0, 0, 0, 0} },

                                                             { {0, 0, 0, 0, 0},
                                                               {0, -1, -1, -1, 0},
                                                               {0, -1, 8, -1, 0},
                                                               {0, -1, -1, -1, 0},
                                                               {0, 0, 0, 0, 0} },

                                                             { {0, 0, 0, 0, 0},
                                                               {0, 0, 0, 0, 0},
                                                               {0, 0, 4, 0, 0},
                                                               {0, 0, 0, 0, 0},
                                                               {0, 0, 0, 0, 0} },

                                                             { {0, 0, 0, 0, 0},
                                                               {0, -1, -1, -1, 0},
                                                               {0, -1, 8, -1, 0},
                                                               {0, -1, -1, -1, 0},
                                                               {0, 0, 0, 0, 0} },
                                                             { {0, 0, 0, 0, 0},
                                                               {0, 0, 0, 0, 0},
                                                               {0, 0, 0, 0, 0},
                                                               {0, 0, 0, 0, 0},
                                                               {0, 0, 0, 0, 0} }
                                                       };

    Filter3D * filter = (Filter3D*) malloc(sizeof(Filter3D));
    // Set filter dimensions
    filter->x = FILTER_SIZE;
    filter->y = FILTER_SIZE;
    filter->z = FILTER_SIZE;
    // Set filter data elements
    for (int z = 0; z < FILTER_SIZE; z++)
        for (int y = 0; y < FILTER_SIZE; y++)
            for (int x = 0; x < FILTER_SIZE; x++) {
                (filter->data)[z][y][x] = data[z][y][x];
            }
    // Set filter factor and bias
    filter->factor = .25;
    filter->bias =0;
    return filter;
}

// Given a z dimension within the video loads in the next INPUT_TILE_Z number
// of frames from the infiles directory
void loadFrames(PPMImage * frames, int z, int totalFrames)
{
    char instr[80];
    // Read in INPUT_TILE_Z number of frames
    for (int i = 0; i < INPUT_TILE_Z; i++)
    {
        int fileNum = i + z + 1 - FILTER_SIZE / 2;
        // Only read in frames that exist
        if (fileNum <= totalFrames && fileNum > 0)
        {
            sprintf(instr, "../infiles/tmp%03d.ppm", fileNum);
            frames[i] = *readPPM(instr);
        }
    }
}

// Given an x, y, and z dimension within the video gets a chunk of pixels from frames to be processed
// by the next kernel call
void getPixels(PPMImage frames[], PPMPixel *data, int x, int y, int z, int width, int height, int depth)
{
    for (int k = 0; k < INPUT_TILE_Z; k++)
    {
        for (int j = 0; j < INPUT_TILE_Y; j++)
        {
            for (int i = 0; i < INPUT_TILE_X; i++)
            {
                int data_x = i + x - FILTER_SIZE / 2;
                int data_y = j + y - FILTER_SIZE / 2;
                int data_z = k + z - FILTER_SIZE / 2;
                // if data is within bounds of the video file write the pixel data
                if ((data_x >= 0) && (data_x < width) && (data_y >= 0) && (data_y < height) &&
                    (data_z >= 0) && (data_z < depth))
                    data[k * INPUT_TILE_X * INPUT_TILE_Y + j * INPUT_TILE_X + i] = frames[k].data[data_y * width + data_x];
                // otherwise write a black pixel (all zeroes)
                else
                {
                    data[k * INPUT_TILE_X * INPUT_TILE_Y + j * INPUT_TILE_X + i].red = 0;
                    data[k * INPUT_TILE_X * INPUT_TILE_Y + j * INPUT_TILE_X + i].blue = 0;
                    data[k * INPUT_TILE_X * INPUT_TILE_Y + j * INPUT_TILE_X + i].green = 0;
                }
            }
        }
    }
}

// Given an x, y, anx z dimension within the video, writes the processed pixels out to memory
void writePixels(PPMPixel * data, PPMImage * frames, int x, int y, int z, int width, int height)
{
    for (int k = 0; k < OUTPUT_TILE_Z; k++)
        for (int j = 0; j < OUTPUT_TILE_Y; j++)
            for (int i = 0; i < OUTPUT_TILE_X; i++) {
                // if pixel is within bounds write it out to memory
                if(x+i < width && y + j < height)
                    frames[k].data[width*(y+j)+ x+i] = data[k * OUTPUT_TILE_X * OUTPUT_TILE_Y + i * OUTPUT_TILE_X + j];
            }
}

// Given a z dimension in the video, writes OUTPUT_TILE_Z number of frames to disk
void writeFrames(PPMImage * frames, int z, int totalFrames)
{
    char outstr[80];
    for (int i = 0; i < OUTPUT_TILE_Z; i++)
    {
        int fileNum = i + z + 1;
        // if frame is within bounds write it out to disk
        if (fileNum <= totalFrames)
        {
            sprintf(outstr, "../outfiles/tmp%03d.ppm", fileNum);
            writePPM(outstr, &(frames[i]));
        }
    }
}

int main(int argc, char *argv[]){
    char *infile = (char*)"foreman.mp4";
    char ffmpegString[200];

    // if argument is given, go to input_videos directory and create 301 frames of that video
    // in the infiles directory
    if(argc > 1) {
      infile = argv[1];
      if (!system(NULL)) {exit (EXIT_FAILURE);}
      system("exec rm -r ../infiles/*");
      sprintf(ffmpegString, "ffmpeg -i ../input_videos/%s -vframes 301 ../infiles/tmp%%03d.ppm", infile);
      system (ffmpegString);
    }

    // remove old frames from outfiles
    system("exec rm -r -f ../outfiles/*");

    int totalFrames = 0;
    DIR * dirp;
    struct dirent * entry;

    // get total number of frames that need to be processed
    dirp = opendir("../infiles");
    while ((entry = readdir(dirp)) != NULL) {
        if (entry->d_type == DT_REG) { /* If the entry is a regular file */
             totalFrames++;
        }
    }

    closedir(dirp);

    // Initialize profiling variables
    clock_t begin, end;
    double time_spent = 0.0;


    PPMPixel *imageData_d, *outputData_d, *outputData_h, *inputData_h;
    Filter3D * filter_h = initializeFilter();

    cudaError_t cuda_ret;

    // Read first frame to get width and height information
    PPMImage *image =  readPPM("../infiles/tmp001.ppm");

    // Malloc space for input and output pixels
    inputData_h  = (PPMPixel *)malloc(INPUT_TILE_X * INPUT_TILE_Y * INPUT_TILE_Z * sizeof(PPMPixel));
    outputData_h = (PPMPixel *)malloc(OUTPUT_TILE_X * OUTPUT_TILE_Y * OUTPUT_TILE_Z * sizeof(PPMPixel));

    // Malloc space for input and output frames
    PPMImage inputFrames[INPUT_TILE_Z], outputFrames[OUTPUT_TILE_Z];
    for (int i = 0; i < INPUT_TILE_Z; i++) {
        inputFrames[i].x = image->x;
        inputFrames[i].y = image->y;
        inputFrames[i].data = (PPMPixel *)malloc(image->x * image->y * sizeof(PPMPixel));
    }
    for (int i = 0; i < OUTPUT_TILE_Z; i++) {
        outputFrames[i].x = image->x;
        outputFrames[i].y = image->y;
        outputFrames[i].data = (PPMPixel *)malloc(image->x * image->y * sizeof(PPMPixel));
    }

    // Malloc device memory for input and outputs
    cuda_ret = cudaMalloc((void**)&(imageData_d), INPUT_TILE_X * INPUT_TILE_Y * INPUT_TILE_Z * sizeof(PPMPixel));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cuda_ret = cudaMalloc((void**)&(outputData_d), OUTPUT_TILE_X * OUTPUT_TILE_Y * OUTPUT_TILE_Z * sizeof(PPMPixel));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    // Copy filter to constant memory on device
    cudaMemcpyToSymbol(filter_c, filter_h, sizeof(Filter3D));
    cudaDeviceSynchronize();

    // Set grid and block dims for tiled input/outputs
    dim3 dim_block(BLOCK_SIZE + FILTER_SIZE / 2, BLOCK_SIZE + FILTER_SIZE / 2, INPUT_TILE_Z);
    dim3 dim_grid((INPUT_TILE_X + 1) / BLOCK_SIZE + 1,
                  (INPUT_TILE_Y + 1) / BLOCK_SIZE + 1,
                  1);

    // Loop over x, y, z dimensions of the video and for each chunk process it with a kernel call
    for (int z = 0; z < totalFrames; z+=OUTPUT_TILE_Z)
    {
        // For each z, load INPUT_TILE_Z frames from disk
        loadFrames(inputFrames, z, totalFrames);
        for (int y = 0; y < image->y; y+=OUTPUT_TILE_Y)
        {
            for (int x = 0; x < image->x; x+=OUTPUT_TILE_X)
            {
                // For each x y z get a chunk of pixels from the input frames
                getPixels(inputFrames, inputData_h, x, y, z, image->x, image->y, totalFrames);
                // Copy pixels to device memory
                cudaMemcpy(imageData_d, inputData_h, INPUT_TILE_X * INPUT_TILE_Y * INPUT_TILE_Z * sizeof(PPMPixel),
                           cudaMemcpyHostToDevice);
                begin = clock();
                // Process pixels with convolution kernel
                convolution<<<dim_grid, dim_block>>>(imageData_d, outputData_d);
                cuda_ret = cudaDeviceSynchronize();
                if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

                end = clock();
                time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
                // Copy processed pixels from device to host
                cudaMemcpy(outputData_h, outputData_d, OUTPUT_TILE_X * OUTPUT_TILE_Y * OUTPUT_TILE_Z * sizeof(PPMPixel),
                           cudaMemcpyDeviceToHost);
                // Write pixels to outputFrames in memory
                writePixels(outputData_h, outputFrames, x, y, z, image->x, image->y);
            }
        }
        // Write output frames to disk
        writeFrames(outputFrames, z, totalFrames);

    }

    // Free device and host memory
    free(inputData_h);
    free(outputData_h);
    for (int i = 0; i < INPUT_TILE_Z; i++)
        free(inputFrames[i].data);
    for (int i = 0; i < OUTPUT_TILE_Z; i++)
        free(outputFrames[i].data);
    cudaFree(imageData_d);
    cudaFree(outputData_d);

    // Combine frames into a single video with ffmpeg
    if (!system(NULL)) { exit (EXIT_FAILURE);}
    sprintf(ffmpegString, "ffmpeg -framerate 24 -i ../outfiles/tmp%%03d.ppm -c:v libx264 -r 30 -pix_fmt yuv420p ../outfilter.mp4");
    system (ffmpegString);

    printf("%f seconds spent\n", time_spent);

}
