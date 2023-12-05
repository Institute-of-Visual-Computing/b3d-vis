#pragma once

#include "cuda_runtime.h"

namespace b3d::unity_cuda_interop
{
	class SyncPrimitive
	{
	public:
		virtual ~SyncPrimitive() {}

		auto isValid() const -> bool
		{
			return isValid_;
		}

		virtual auto importToCUDA() -> void = 0;

		auto getCudaSemaphore() const -> cudaExternalSemaphore_t
		{
			return cudaSemaphore_;
		}

	protected:
		SyncPrimitive() {}

		bool isValid_{ false };

		cudaExternalSemaphore_t cudaSemaphore_{ nullptr };
	};
}// namespace b3d::unity_cuda_interop
