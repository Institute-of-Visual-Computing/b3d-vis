#pragma once
#include <d3d11.h>
#include <d3d11_3.h>

#include "SyncPrimitive.h"

namespace b3d::unity_cuda_interop {

	class PluginLogger;
	class SyncPrimitiveD3D11 final : public SyncPrimitive
	{
	public:
		SyncPrimitiveD3D11(PluginLogger* logger, ID3D11Device* d3d11Device);

		~SyncPrimitiveD3D11() override;

		auto importToCUDA() -> void override;

		auto getD3D11Fence() const -> ID3D11Fence*
		{
			return d3d11Fence_;
		}

	protected:

	private:
		ID3D11Device* d3d11Device_{ nullptr };
		ID3D11Fence* d3d11Fence_{ nullptr };
		HANDLE d3d11SharedFenceHandle_{ nullptr };

		PluginLogger *logger_{ nullptr };
	};
} // namespace b3d::unity_cuda_interop
