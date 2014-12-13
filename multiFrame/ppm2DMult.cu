#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include "ppmKernel.cu"
#include "ppm.h"

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

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

void changeColorPPM(PPMImage *img)
{
    int i;
    if(img){

         for(i=0;i<img->x*img->y;i++){
              int avg = (img->data[i].red + img->data[i].green + img ->data[i].blue) / 3;

              img->data[i].red = avg;
              img->data[i].green = avg;
              img->data[i].blue = avg;
         }
    }
}

Filter * initializeFilter()
{
    int data[FILTER_SIZE * FILTER_SIZE] = {0, 0, 0, 0, 0,
                                           0, -1, -1, -1, 0,
                                           0, -1, 8, -1, 0,
                                           0, -1, -1, -1, 0,
                                           0, 0, 0, 0, 0};
    Filter * filter = (Filter*) malloc(sizeof(Filter));
    filter->x = FILTER_SIZE;
    filter->y = FILTER_SIZE;
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++)
       filter->data[i] = data[i];
    filter->factor = 1.0;
    filter->bias =0;
    return filter;
}

int main(int argc, char *argv[]){

  char *infile = (char*)"foreman.mp4";
  char ffmpegString[200];
  if(argc > 1) {
    infile = argv[1];

    if (!system(NULL)) {exit (EXIT_FAILURE);}
    system("exec rm -r ../infiles/*");
    sprintf(ffmpegString, "ffmpeg -i ../input_videos/%s -f image2 -vf fps=fps=24 ../infiles/tmp%%03d.ppm", infile);
    system (ffmpegString);

  }
  system("exec rm -r -f ../outfiles/*");

  int totalFrames = 0;
  DIR * dirp;
  struct dirent * entry;

  dirp = opendir("../infiles"); /* There should be error handling after this */
  while ((entry = readdir(dirp)) != NULL) {
      if (entry->d_type == DT_REG) { /* If the entry is a regular file */
           totalFrames++;
      }
  }
  totalFrames -= 1;
  closedir(dirp);
  printf("%d\n", totalFrames);
    clock_t begin, end;
    double time_spent = 0.0;


    /* here, do your time-consuming job */

    char instr[80];
    char outstr[80];
    int i = 0;


    PPMImage images[totalFrames];

    PPMPixel *imageData_d, *outputData_d, *outputData_h, *imageData_h;
    Filter * filter_h = initializeFilter();

    cudaError_t cuda_ret;

    // read input frames into memory
    for(i = 0; i < totalFrames; i++) {
        sprintf(instr, "../infiles/tmp%03d.ppm", i+1);
        images[i] = *readPPM(instr);
    }
    PPMImage *image;
    image = &images[0];

    // malloc host memory
    outputData_h = (PPMPixel *)malloc(IMAGES_PER_BLOCK*image->x*image->y*sizeof(PPMPixel));
    imageData_h = (PPMPixel *)malloc(IMAGES_PER_BLOCK*image->x*image->y*sizeof(PPMPixel));

    // malloc device memory
    cuda_ret = cudaMalloc((void**)&(imageData_d),IMAGES_PER_BLOCK*image->x*image->y*sizeof(PPMPixel));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cuda_ret = cudaMalloc((void**)&(outputData_d),IMAGES_PER_BLOCK*image->x*image->y*sizeof(PPMPixel));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    // set the output image's properties
    PPMImage *outImage;
    outImage = (PPMImage *)malloc(sizeof(PPMImage));
    outImage->x = image->x;
    outImage->y = image->y;

    //setup shared memory
    cudaMemcpyToSymbol(filter_c, filter_h, sizeof(Filter));
    cudaDeviceSynchronize();

    for(i = 0; i < totalFrames; i+=IMAGES_PER_BLOCK) {

        // load IMAGES_PER_BLOCk images into a flat pixel array
        for(int x = 0; x < IMAGES_PER_BLOCK; x++) {
          if(i+x < totalFrames) {
            image = &images[i+x];
            for(int j=0; j < image->x*image->y; j++) {
              imageData_h[image->x*image->y*x + j] = (image->data)[j];
            }
          }
        }

      // copy the image data to the device
      cuda_ret = cudaMemcpy(imageData_d, imageData_h, IMAGES_PER_BLOCK*image->x*image->y*sizeof(PPMPixel), cudaMemcpyHostToDevice);
      if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device");

      // Convolution
      const unsigned int grid_x = (image->x - 1) / OUTPUT_TILE_SIZE + 1;
      const unsigned int grid_y = (image->y -1) / OUTPUT_TILE_SIZE + 1;
      dim3 dim_grid(grid_x, grid_y, 1);
      dim3 dim_block(INPUT_TILE_SIZE, INPUT_TILE_SIZE, 1);
      
      begin = clock();

      // Convolution
      convolution<<<dim_grid, dim_block>>>(imageData_d, outputData_d, image->x, image->y, IMAGES_PER_BLOCK);        

      // Black and white
      // dim3 dim_grid, dim_block;
      // dim_grid = dim3(image->y, 1,1);
      // dim_block = dim3(image->x, 1, 1);
      // blackAndWhite<<<dim_grid, dim_block>>>(imageData_d, outputData_d, image->x, image->y, IMAGES_PER_BLOCK);

      cuda_ret = cudaDeviceSynchronize();
      if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
      end = clock();
      time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

      // the processed image data from device 
      cuda_ret = cudaMemcpy(outputData_h, outputData_d,IMAGES_PER_BLOCK*image->x*image->y*sizeof(PPMPixel), cudaMemcpyDeviceToHost);
      if(cuda_ret != cudaSuccess) FATAL("Unable to copy to host");

      // extract processed image data from the flattened array and write the image to disk
      for(int x = 0; x < IMAGES_PER_BLOCK; x++) {
        if(i+x < totalFrames) {
          sprintf(outstr, "../outfiles/tmp%03d.ppm", i+x+1);
          outImage->data = &(outputData_h[x*image->x*image->y]);
          writePPM(outstr,outImage);

        }
      }
    }

    free(outputData_h);
    free(outImage);
    cudaFree(imageData_d);
    cudaFree(outputData_d);



    if (!system(NULL)) { exit (EXIT_FAILURE);}
    sprintf(ffmpegString, "ffmpeg -framerate 24 -i ../outfiles/tmp%%03d.ppm -c:v libx264 -r 30 -pix_fmt yuv420p outfilter.mp4");
    system (ffmpegString);

    printf("%f seconds spent\n", time_spent);


}