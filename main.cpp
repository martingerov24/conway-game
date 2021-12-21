#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "build/stb_image.h"
#include <vector>

std::vector<uint8_t> readFile(char* fileName, int32_t& height, int32_t& width)
{
	int32_t channels;
	unsigned char* rawImage = stbi_load(fileName, &width, &height, &channels, 1);
	if (rawImage == NULL) { throw "the input image is null"; }

	std::vector<uint8_t> image(height * width);
	memcpy(&image[0], rawImage, image.size());
	if (image.empty()) { throw "vector was not filled with data"; }

	stbi_image_free(rawImage);
	delete[] fileName;
	return image;
}


int main() {
	printf("hello world!");
	return 0;
}
