#pragma once
namespace b3d::unity_cuda_interop
{
	struct UnityInputTexture
	{
		void* pointer{ nullptr };
		renderer::Extent extent{ 0, 0, 0 };
	};

	struct NativeTextureData
	{
		UnityInputTexture colorTexture{};
		UnityInputTexture depthTexture{};
	};
}
