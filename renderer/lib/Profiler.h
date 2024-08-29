#pragma once

#include <ProfilerTask.h>
#include <vector>

#include "ProfilerResult.h"
namespace b3d::profiler
{
	class Profiler final
	{
	public:
		auto collectGpuTimers(const std::vector<ProfilerResult>& timers) -> void;
		auto collectCpuTimers(const std::vector<ProfilerResult>& timers) -> void;

		[[nodiscard]] auto gpuProfilerTasks() -> std::vector<legit::ProfilerTask>&
		{
			return gpuTasks_;
		}
		[[nodiscard]] auto cpuProfilerTasks() -> std::vector<legit::ProfilerTask>&
		{
			return cpuTasks_;
		}

		auto nextFrame() -> void
		{
			gpuOffset_ = 0.0f;
			cpuOffset_ = 0.0f;

			cpuTasks_.clear();
			gpuTasks_.clear();
		}

	private:
		std::vector<legit::ProfilerTask> gpuTasks_{};
		std::vector<legit::ProfilerTask> cpuTasks_{};

		float gpuOffset_{ 0.0f };
		float cpuOffset_{ 0.0f };
	};
} // namespace b3d::profiler
