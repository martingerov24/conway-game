#pragma once 
#include <vector>
class Game
{
private:
	void deathPixel(int currentPixel)
	{
		int sum = image[currentPixel - 1] +
			image[currentPixel + 1] +
			image[currentPixel - width] +
			image[currentPixel + width] +
			image[currentPixel + width + 1] +
			image[currentPixel + width - 1] +
			image[currentPixel - width + 1] +
			image[currentPixel - width - 1];
		if (sum == 3)
		{
			m_resultingImage[currentPixel] = 1;
			return;
		}
	}
	void livingPixel(int currentPixel)
	{
		int fst = image[currentPixel - width - 1] & 0x1;

		int sum =  (image[currentPixel - 1] & 0x1) +
			(image[currentPixel + 1] & 0x1 )	   +
			(image[currentPixel - width] & 0x1)    +
			(image[currentPixel + width] & 0x1)   +
			(image[currentPixel + width + 1] & 0x1)+
			(image[currentPixel + width - 1] & 0x1)+
			(image[currentPixel - width + 1] & 0x1)+
			(fst);  // this is sum of all the neightbors from top
					//left to bottom right
		/*assert(currentPixel + width + 1 <= image.size());
		assert(currentPixel - width - 1 > 0);*/
		if (sum < 2)
		{
			m_resultingImage[currentPixel] = 0;
		}
		else if (sum == 2 || sum == 3)
		{
			return;
		}
		else
		{
			m_resultingImage[currentPixel] = 0;
		}
	}
public:
	Game(std::vector<uint8_t> &image, const int height, const int width)
		:height(height)
		,width(width)
		,image(image)
	{
		m_resultingImage.resize(width * height);
	}
	void imageIteration()
	{

		int calculations = 0;
		int currentPixel = 0;
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				currentPixel = i * width + j;
				if(image[currentPixel == 1])
				{
					livingPixel(currentPixel);
				}
				else// death
				{
					deathPixel(currentPixel);
				}
			}
		}

		image = m_resultingImage;
	}
private:
	int height;
	int width;
	std::vector<uint8_t>& image; 
	std::vector<uint8_t> m_resultingImage;
};