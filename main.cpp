#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb_image.h"
#if DEBUG == 0
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stbi_write.h"
#endif

#include <string>
#include <vector>
#include "../external/Game.h"
#include "../external/threadPool.h"
#include "../external/GameCuda.h"


#if DEBUG==0
void debugWrite(std::vector<uint8_t>& img, int32_t height, int32_t width, int32_t channels = 1)
{
	stbi_write_png("sky.png", width, height, channels, img.data(), width * channels); // stbi is deleting the pointer
}
#endif 

std::vector<uint8_t> readFile(char* fileName, int32_t& height, int32_t& width)
{
	int32_t channels = 1;
	unsigned char* rawImage = stbi_load(fileName, &width, &height, &channels, 1);
	if (rawImage == NULL) { throw "the input image is null"; }

	std::vector<uint8_t> image(height * width );
	memcpy(&image[0], rawImage, image.size());
	if (image.empty()) { throw "vector was not filled with data"; }

	stbi_image_free(rawImage);
	return image;
}
std::vector<uint8_t> readingP1(char*& fileName, int& height, int& width) // not used
{
	FILE* rdFile = fopen(fileName, "rb+");
	if (rdFile == 0) { throw "the file is was not found!"; }
	char buff[10];
	fscanf(rdFile, "%s\n", buff);
	fscanf(rdFile, "%d %d\n", &width, &height);
	std::vector<uint8_t> data(height * width);
	fread(reinterpret_cast<char*>(&data[0]), 1, height * width, rdFile);
	//rdFile.read(reinterpret_cast<char*>(&data[0]), height * width * sizeof(uint16_t));
	fclose(rdFile);
	return data;
}

void bindTexture(GLuint texture)
{
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
}
void onNewFrame()
{
	glfwPollEvents();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}
void createContext(GLFWwindow* window)
{
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
}

void loop() // this was used for cuda, but i will send you the code later this week
{
	int width; int height;


	bool ifP1 = 0; // this is mainly used for debug, in smaller images, like what you provided
	//that's why i did not do it in separated cuda function


	std::vector<uint8_t> image;
	if (ifP1)
	{
		char* filename = "../image.pbm";
		image = readingP1(filename, height, width);
		for (int i = 0; i < image.size(); i++)
		{
			image[i] *= 255;
		}
	}
	else
	{
		char* filename = "../conwayGame.png";
		image = readFile(filename, height, width);
	}
		std::vector<uint8_t> result(image.size());
	/// <for Cuda>
	/// to be fully async i provide on each use stream, where the code will be executed, 
	/// but if you provide different one, data won't be malloced 
	/// </for Cuda>
	cudaError_t cudaStatus = cudaError_t(0);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaStatus = cudaSetDevice(0);
	assert(cudaStatus == cudaSuccess && "you do not have cuda capable device!");
	cudaStream_t stream;
	cudaStatus = cudaStreamCreate(&stream);
	assert(cudaStatus == cudaSuccess && "you do not have cuda capable device!");

	CudaGame game(image, height, width);
	game.memoryAllocationAsyncOnDevice(stream,cudaStatus); // this is the malloc for cuda
	game.cudaUploadImage(stream, cudaStatus);
	game.kernel(stream);
	game.downloadAsync(stream, result, image.size(), cudaStatus);


	//untill wile loop, there won't be anything interesting, but only opengl code
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	if (!glfwInit()) {
		throw "glfwInit() FAILED!";
	}
	GLFWwindow* window = glfwCreateWindow(800, 600, "Raw-File Viewer", NULL, NULL);

	if (!window) {
		glfwTerminate();
		throw "no window created";
	}
	createContext(window);
	game.sync(stream, cudaStatus);

	bool is_show = true;
	GLuint texture;
	glGenTextures(1, &texture);



	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);// super special, 1 day of debugging until i found that opengl by default have padding
	while (!glfwWindowShouldClose(window))
	{
		//game.cudaUploadImage(stream, cudaStatus);
		//game.kernel(stream);
		//game.downloadAsync(stream, result, image.size(), cudaStatus);
		onNewFrame();
		ImGui::Begin("raw Image", &is_show);
		bindTexture(texture);
		//game.sync(stream, cudaStatus);
		//game.sync(stream, cudaStatus);
		image = result;
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, image.data());
		ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture)), ImVec2(1024, 512));
		ImGui::End();
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glDeleteTextures(sizeof(texture), &texture);
		glfwSwapBuffers(window);
	}

	ImGui_ImplGlfw_Shutdown();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
	glfwDestroyWindow(window);

	game.cudaFreeAcync(stream, cudaStatus);
}
int main() {
	loop(); // loop is the Start function, where all thins are invocked
	return 0;
}
