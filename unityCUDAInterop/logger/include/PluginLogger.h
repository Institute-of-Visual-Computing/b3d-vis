#pragma once

#include <source_location>

#include "IUnityLog.h"

namespace b3d::unity_cuda_interop
{
	class PluginLogger
	{
	  public:
		explicit PluginLogger(IUnityLog* unityLog);

		~PluginLogger() = default;

		void log(const char* message, std::source_location location = std::source_location::current()) const;

	  private:
		IUnityLog* unityLog_{ nullptr };
	};
} // namespace b3d::unity_cuda_interop
