#include "PluginLogger.h"

using namespace b3d::unity_cuda_interop;

PluginLogger::PluginLogger(IUnityLog* unityLog) : unityLog_(unityLog)
{
}

auto PluginLogger::log(const char* message, const std::source_location location) const -> void
{
	unityLog_->Log(UnityLogType::kUnityLogTypeLog, message, location.file_name(), location.line());
}
