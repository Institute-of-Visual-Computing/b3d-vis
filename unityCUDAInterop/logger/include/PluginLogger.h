#pragma once

#include <source_location>

#include "IUnityInterface.h"
struct IUnityLog;

namespace b3d::unity_cuda_interop
{
	class PluginLogger
	{
	  public:
		PluginLogger(IUnityLog* unityLog);

		// ReSharper disable once CppEnforceFunctionDeclarationStyle
		UNITY_INTERFACE_EXPORT void log(const char* message,
		                                std::source_location location = std::source_location::current()) const;

	  private:
		IUnityLog* unityLog_{ nullptr };
	};
} // namespace b3d::unity_cuda_interop
