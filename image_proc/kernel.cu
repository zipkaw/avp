#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "transforms.cu"
#include "recover.cu"

int main()
{
    int width, height, numComp;
    unsigned char *image = stbi_load("/home/plantator/AVP/image_proc/image.png", &width, &height, &numComp, 0);

    uint8_t *dev_image, *output_dev_image;
    size_t im_pitch, out_pitch;
    cudaMallocPitch(&dev_image, &im_pitch, width * numComp * sizeof(unsigned char), height);
    cudaMallocPitch(&output_dev_image, &out_pitch, width * numComp * sizeof(unsigned char), height);
    cudaMemcpy2D(dev_image, im_pitch, image, width * numComp * sizeof(unsigned char), width * numComp * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

    const dim3 GRID_DIM{(height + BLOCK_DIM.x - 1) / (BLOCK_DIM.x),
                        (width + BLOCK_DIM.y - 1) / BLOCK_DIM.y};
    ;

    unsigned int *accum = new unsigned int[min(width, height)];
    memset(accum, 0, sizeof(unsigned int) * min(width, height));
    unsigned int *dev_accum = new unsigned int[min(width, height)];
    cudaMalloc(&dev_accum, min(width, height) * sizeof(unsigned int));
    cudaMemcpy(dev_accum, accum, min(width, height) * sizeof(unsigned int), cudaMemcpyHostToDevice);

    hought_transform<<<GRID_DIM, BLOCK_DIM>>>(dev_image, width, height, numComp, im_pitch, dev_accum);
    cudaMemcpy(accum, dev_accum, min(width, height) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    const int radius = collect_accumulator(accum, min(width, height));
    const float coeff = calc_coefficient(width, height, radius);

    bool *mask = new bool[width * height];
    bool *dev_mask;
    memset(mask, false, sizeof(bool) * width * height);
    cudaMalloc(&dev_mask, width * height * sizeof(bool));
    cudaMemcpy(dev_mask, mask, width * height * sizeof(bool), cudaMemcpyHostToDevice);
    fish_eye<<<GRID_DIM, BLOCK_DIM, sizeof(RGBA_t) * BLOCK_DIM.x * BLOCK_DIM.y>>>(dev_image,
                                                                                  output_dev_image,
                                                                                  width, height,
                                                                                  numComp, radius,
                                                                                  coeff, out_pitch,
                                                                                  im_pitch, dev_mask);

    recover<<<GRID_DIM, BLOCK_DIM, sizeof(RGBA_t) * (BLOCK_DIM.x+2) * (BLOCK_DIM.y+2)>>>(output_dev_image,
                                                                                 width, height, numComp,
                                                                                 out_pitch, dev_mask);
    cudaMemcpy(mask, dev_mask, width * height * sizeof(bool), cudaMemcpyDeviceToHost);

    unsigned char *host_image = (unsigned char *)malloc(width * numComp * sizeof(unsigned char) * height);
    cudaMemcpy2D(host_image, width * sizeof(unsigned char) * numComp, output_dev_image, width * sizeof(unsigned char) * numComp, width * sizeof(unsigned char) * numComp, height, cudaMemcpyDeviceToHost);
    stbi_write_png("/home/plantator/AVP/image_proc/image3.png", width, height, 4, host_image, width * 4);
    stbi_image_free(image);
}
