#pragma once

#include "cuda_runtime.h"

class RenderAPI;
struct cudaGraphicsResource;

class Texture
{
public:
	Texture() = delete;

	virtual ~Texture()
	{
		if(cudaGraphicsResource_ != nullptr)
		{
			cudaGraphicsResource_ = nullptr;
		}
		unityNativeTexturePointer_ = nullptr;
		isValid_ = false;
	}

	int getWidth() const { return width_;  }
	int getHeight() const { return height_;  }
	int getDepth() const { return depth_;  }
	bool isValid() const { return isValid_; }

	virtual void registerCUDA() = 0;
	virtual void unregisterCUDA();

	void* getUnityNativeTexturePointer() const { return unityNativeTexturePointer_; }
	cudaGraphicsResource_t getCudaGraphicsResource() const { return cudaGraphicsResource_; }

private:
	void *unityNativeTexturePointer_{ nullptr };

protected:
	Texture(void* unityNativeTexturePointer) : unityNativeTexturePointer_(unityNativeTexturePointer) {}

	int width_{ 0 };
	int height_{ 0 };
	int depth_{ 0 };

	bool isValid_{ false };

	cudaGraphicsResource_t cudaGraphicsResource_{ nullptr };
};
