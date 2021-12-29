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

	std::vector<uint8_t> image(height * width);
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
	char* filename = "../image.pbm";
	int width; int height;
	std::vector<uint8_t> image = readingP1(filename, height, width);

	Game game(image, height, width);
	game.imageIteration();



	debugWrite(image, height, width);
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

	bool is_show = true;
	GLuint texture;
	glGenTextures(1, &texture);

	while (!glfwWindowShouldClose(window))
	{
		onNewFrame();
		ImGui::Begin("raw Image", &is_show);
		bindTexture(texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, image.data());
		ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture)), ImVec2(width, height));
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
}
int main() {
	loop();
	return 0;
}
