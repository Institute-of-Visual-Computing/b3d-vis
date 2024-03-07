#pragma once

#include "RenderData.h"
#include "RenderFeature.h"

namespace b3d::renderer
{

	class RenderSyncFeature final : public RenderFeature
	{
	public:
		explicit RenderSyncFeature(const std::string& name) : RenderFeature{ name }
		{
		}

		auto beginUpdate() -> void override;
		auto endUpdate() -> void override;

	private:
		bool skipUpdate{ false };

		Synchronization* synchronization_{ nullptr };

		cudaTextureObject_t colorMapCudaTexture_{};

		int selectedColoringMode_{ 0 };
		int selectedColoringMap_{ 0 };
	};

} // namespace b3d::renderer
