#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

// Filter struct. See http://lodev.org/cgtutor/filtering.html for more info
typedef struct {
     int x, y;
     double data[25];
     double factor;
     double bias;
} Filter;

#define CREATOR "DA BROS"
#define RGB_COMPONENT_COLOR 255

static PPMImage *readPPM(const char *filename)
{
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              // fprintf(stderr, "Hey! Unable to open file '%s'\n", filename);
              // exit(1);
            return NULL;
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
        while (getc(fp) != '\n')
            ;
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

int print = 0;
// filter a single PPM image
void filterPPM(PPMImage *img, Filter f)
{
    if(!img)
        return;

    int img_x, img_y;
    int f_x, f_y;
    int pixel_x, pixel_y;

    // loop over all pixels in image
    for(img_x = 0; img_x < img->x; img_x++) {
        for(img_y = 0; img_y < img->y; img_y++) {
            // accumulate red, green, and blue vals for this pixel
            int red = 0, green = 0, blue = 0;
            
            // loop over surrounding pixels
            for(f_x = 0; f_x < f.x; f_x++) {
                for(f_y = 0; f_y < f.y; f_y++) {
                    // get absolute locations of surrounding pixels
                    pixel_x = img_x + (f_x - f.x / 2);
                    pixel_y = img_y + (f_y - f.y / 2);

                    // check to make sure pixel_x and pixel_y are valid coordinates
                    if(pixel_x >= 0 && pixel_x < img->x && pixel_y >= 0 && pixel_y < img->y) {
                        red += img->data[pixel_y * img->y + pixel_x].red * f.data[f_y * f.x + f_x];
                        green += img->data[pixel_y * img->y + pixel_x].green * f.data[f_y * f.x + f_x];
                        blue += img->data[pixel_y * img->y + pixel_x].blue * f.data[f_y * f.x + f_x];
                    }
                }
            }

            
            // apply filter factor and bias, and write resultant pixel back to img
            img->data[img_y * img->y + img_x].red = fmin( fmax( (int)(f.factor * red + f.bias), 0), 255);
            img->data[img_y * img->y + img_x].green = fmin( fmax( (int)(f.factor * green + f.bias), 0), 255);
            img->data[img_y * img->y + img_x].blue = fmin( fmax( (int)(f.factor * blue + f.bias), 0), 255);

            if(print < 30) {
                printf("img: %d, %d, %d\n", img->data[img_y * img->y + img_x].red, img->data[img_y * img->y + img_x].green, img->data[img_y * img->y + img_x].blue);
                print++;
            }
        }
    }
}

void freeImage(PPMImage * image) {
    free(image->data);
    free(image);
}

// ppm <stride_len> (<max_frames>)
int main(int argc, char *argv[]){
    if ( argc < 2 ) /* argc should be at least 2 for correct execution */
    {
        /* We print argv[0] assuming it is the program name */
        printf("usage: %s <stride_len> (<max_frames>)\n", argv[0]);
        return -1;
    }

    char instr[80];
    char outstr[80];
    int i = 0, j = 0;
    int numFrames = 301;
    int stride_len = atoi(argv[1]);
    int numChunks = numFrames / stride_len;
    if(numFrames % stride_len) numChunks++;

    clock_t begin, end;
    double time_spent[numChunks + 1];   // time spent for each chunk, plus accumulate total at end
    time_spent[numChunks - 1] = 0;      // init accumulator to 0

    PPMImage images[stride_len];
    Filter blur = {
        .x = 5,
        .y = 5,
        .data = {
            0, 0, 1, 0, 0,
            0, 1, 1, 1, 0,
            1, 1, 1, 1, 1,
            0, 1, 1, 1, 0,
            0, 0, 1, 0, 0
        },
        .factor = 0.25,
        .bias = 0
    };

    // loop over all frames, in chunks of stride_len
    for(i = 0; i < numChunks; i ++) {

        // read 1 chunk of stride_len frames into images[]
        for(j = 0; j < stride_len && j + i*stride_len < numFrames; j++) {
            sprintf(instr, "infiles/baby%03d.ppm", i*stride_len + j + 1);
            PPMImage * img = readPPM(instr);
            if(img == NULL) {
                printf("All files processed\n");
                return 0;
            }
            images[j] = *img;
        }
        
        begin = clock();

        // process chunk of frames in images[]
        for(j = 0; j < stride_len && j + i*stride_len < numFrames; j++) {
            filterPPM(&images[j], blur);
        }
        
        end = clock();
        time_spent[i] = (double)(end - begin) / CLOCKS_PER_SEC;
        time_spent[numChunks - 1] += time_spent[i];

        // write 1 chunk of stride_len frames from images[]
        for(j = 0; j < stride_len && j + i*stride_len < numFrames; j++) {
            sprintf(outstr, "outfiles/baby%03d.ppm", i*stride_len + j + 1);
            writePPM(outstr, &images[j]);
        }
        
        freeImage(&image[j]);
    }

    printf("%f seconds spent\n", time_spent[numChunks - 1]);

    return 0;
}