#include "Profiler.h"

#include <array>

namespace
{
	constexpr std::array colors = { legit::Colors::turqoise,  legit::Colors::greenSea,	  legit::Colors::emerald,
									legit::Colors::nephritis, legit::Colors::peterRiver,  legit::Colors::belizeHole,
									legit::Colors::amethyst,  legit::Colors::wisteria,	  legit::Colors::sunFlower,
									legit::Colors::orange,	  legit::Colors::carrot,	  legit::Colors::pumpkin,
									legit::Colors::alizarin,  legit::Colors::pomegranate, legit::Colors::clouds,
									legit::Colors::silver };
}

auto b3d::profiler::Profiler::collectGpuTimers(const std::vector<ProfilerResult>& timers) -> void
{
	const auto& currentTimings = timers;
	auto gpuLastEndTime = 0.0f;
	for (const auto& [name, start, stop] : currentTimings)
	{
		auto profilerTask = legit::ProfilerTask{};
		profilerTask.name = name;
		profilerTask.startTime = gpuOffset_ + start / 1000.0f;
		profilerTask.endTime = gpuOffset_ + stop / 1000.0f;

		const auto hash = std::hash<std::string_view>{}(name);

		profilerTask.color = colors[hash % colors.size()];
		gpuTasks_.push_back(profilerTask);

		gpuLastEndTime = std::max(gpuLastEndTime, static_cast<float>(profilerTask.endTime));
	}
	gpuOffset_ += gpuLastEndTime;
}
auto b3d::profiler::Profiler::collectCpuTimers(const std::vector<ProfilerResult>& timers) -> void
{
	const auto& currentTimings = timers;
	auto cpuLastEndTime = 0.0f;
	for (const auto& [name, start, stop] : currentTimings)
	{
		auto profilerTask = legit::ProfilerTask{};
		profilerTask.name = name;
		profilerTask.startTime = cpuOffset_ + start / 1000.0f;
		profilerTask.endTime = cpuOffset_ + stop / 1000.0f;
		const auto hash = std::hash<std::string_view>{}(name);
		profilerTask.color = colors[hash % colors.size()];
		cpuTasks_.push_back(profilerTask);

		cpuLastEndTime = std::max(cpuLastEndTime, static_cast<float>(profilerTask.endTime));
	}
	cpuOffset_ += cpuLastEndTime;
}
