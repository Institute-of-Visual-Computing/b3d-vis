#pragma once

#include <chrono>
#include <sstream>
#include <string>

#include "ImGuiProfilerRenderer.h"

class ProfilersWindow final
{
public:
	ProfilersWindow(const ProfilersWindow& other) = default;
	ProfilersWindow(ProfilersWindow&& other) noexcept = default;
	auto operator=(ProfilersWindow other) -> ProfilersWindow&
	{
		using std::swap;
		swap(*this, other);
		return *this;
	}
	ProfilersWindow();
	auto render() -> void;

private:
	bool stopProfiling_;
	int frameOffset_;
	ImGuiUtils::ProfilerGraph cpuGraph_;
	ImGuiUtils::ProfilerGraph gpuGraph_;

public:
	[[nodiscard]] auto isProfiling() const -> bool
	{
		return stopProfiling_;
	}
	[[nodiscard]] auto gpuGraph() -> ImGuiUtils::ProfilerGraph&
	{
		return gpuGraph_;
	}

private:
	int frameWidth_;
	int frameSpacing_;
	bool useColoredLegendText_;
	using TimePoint = std::chrono::time_point<std::chrono::system_clock>;
	TimePoint prevFpsFrameTime_;
	size_t fpsFramesCount_;
	float avgFrameTime_;
};
inline ProfilersWindow::ProfilersWindow() : cpuGraph_(300), gpuGraph_(300)
{
	stopProfiling_ = false;
	frameOffset_ = 0;
	frameWidth_ = 3;
	frameSpacing_ = 1;
	useColoredLegendText_ = true;
	prevFpsFrameTime_ = std::chrono::system_clock::now();
	fpsFramesCount_ = 0;
	avgFrameTime_ = 1.0f;
}
inline auto ProfilersWindow::render() -> void
{
	fpsFramesCount_++;
	{
		const auto currFrameTime = std::chrono::system_clock::now();
		const auto fpsDeltaTime = std::chrono::duration<float>(currFrameTime - prevFpsFrameTime_).count();
		if (fpsDeltaTime > 0.5f)
		{
			this->avgFrameTime_ = fpsDeltaTime / static_cast<float>(fpsFramesCount_);
			fpsFramesCount_ = 0;
			prevFpsFrameTime_ = currFrameTime;
		}
	}

	std::stringstream title;
	title.precision(2);
	title << std::fixed << "Legit profiler [" << 1.0f / avgFrameTime_ << "fps\t" << avgFrameTime_ * 1000.0f
		  << "ms]###ProfilerWindow";
	
	ImGui::Begin(title.str().c_str(), nullptr, ImGuiWindowFlags_NoScrollbar);
	const auto canvasSize = ImGui::GetContentRegionAvail();

	const auto sizeMargin = static_cast<int>(ImGui::GetStyle().ItemSpacing.y);
	constexpr auto maxGraphHeight = 200;
	const auto availableGraphHeight = (static_cast<int>(canvasSize.y) - sizeMargin); // / 2;
	const auto graphHeight = std::min(maxGraphHeight, availableGraphHeight);
	constexpr auto legendWidth = 400;
	const auto graphWidth = static_cast<int>(canvasSize.x) - legendWidth;
	gpuGraph_.RenderTimings(graphWidth, legendWidth, graphHeight, frameOffset_);

	if (!stopProfiling_)
	{
		frameOffset_ = 0;
	}
	gpuGraph_.frameWidth = frameWidth_;
	gpuGraph_.frameSpacing = frameSpacing_;
	gpuGraph_.useColoredLegendText = useColoredLegendText_;
	cpuGraph_.frameWidth = frameWidth_;
	cpuGraph_.frameSpacing = frameSpacing_;
	cpuGraph_.useColoredLegendText = useColoredLegendText_;

	ImGui::End();
}
