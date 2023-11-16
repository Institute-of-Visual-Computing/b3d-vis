#pragma once

#include "cuda_runtime.h"

namespace b3d::unity_cuda_interop::runtime
{
	class RenderAPI;
}

// ReSharper disable once CppInconsistentNaming
struct cudaGraphicsResource;

namespace b3d::unity_cuda_interop
{
	class Texture
	{
	public:
		Texture() = delete;

		virtual ~Texture()
		{
			if (cudaGraphicsResource_ != nullptr)
			{
				cudaGraphicsResource_ = nullptr;
			}
			unityNativeTexturePointer_ = nullptr;
			isValid_ = false;
		}

		auto getWidth() const -> int
		{
			return width_;
		}
		auto getHeight() const -> int
		{
			return height_;
		}
		auto getDepth() const -> int
		{
			return depth_;
		}
		auto isValid() const -> bool
		{
			return isValid_;
		}

		virtual auto registerCUDA() -> void = 0;
		virtual auto unregisterCUDA() -> void;

		auto getUnityNativeTexturePointer() const -> void*
		{
			return unityNativeTexturePointer_;
		}
		auto getCudaGraphicsResource() const -> cudaGraphicsResource_t
		{
			return cudaGraphicsResource_;
		}

	private:
		void* unityNativeTexturePointer_{ nullptr };

	protected:
		Texture(void* unityNativeTexturePointer) : unityNativeTexturePointer_(unityNativeTexturePointer)
		{
		}

		int width_{ 0 };
		int height_{ 0 };
		int depth_{ 0 };

		bool isValid_{ false };

		cudaGraphicsResource_t cudaGraphicsResource_{ nullptr };
	};
} // namespace b3d::unity_cuda_interop
