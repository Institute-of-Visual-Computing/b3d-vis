#pragma once

#include "RenderingContext.h"
#include "SyncPrimitive/SyncPrimitive_D3D11.h"

namespace b3d::unity_cuda_interop
{
	class PluginLogger;
}

namespace b3d::unity_cuda_interop::runtime
{
	class RenderingContext_D3D11 final : public RenderingContext
	{
		public:
			RenderingContext_D3D11(PluginLogger* logger, ID3D11Device* d3d11Device);
			auto signal(SyncPrimitive* syncPrimitive, unsigned value) -> void override;
			auto wait(SyncPrimitive* syncPrimitive, unsigned value) -> void override;

		protected:

	private:
		ID3D11Device* d3d11Device_{ nullptr };
		PluginLogger* logger_{ nullptr };

	};
} // namespace b3d::unity_cuda_interop::runtime
