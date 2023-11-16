#pragma once

namespace b3d::unity_cuda_interop::runtime
{
	enum class PluginRenderEventTypes : int
	{
		rteInitialize = 0,
		rteTeardown,
		actionRegister,
		actionUnregister,

		actionRenderEventTypesMax
	};
} // namespace b3d::unity_cuda_interop::runtime
