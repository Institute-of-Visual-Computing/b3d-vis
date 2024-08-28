#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include <cassert>

#include "GLUtils.h"
#include "ProfilerResult.h"

template <int PoolSize, int DoubleBufferedFrames = 2, bool WaitOnNotReady = false>
class GlGpuTimers final
{
public:
	struct Timings
	{
		GLuint64 start;
		GLuint64 stop;
	};

	class Record
	{
	public:
		auto start() const -> void;
		auto stop() const -> void;

	private:
		friend GlGpuTimers;

		Record(GlGpuTimers* parent, const GLuint64 start, const GLuint64 stop)
			: parent_{ parent }, start_{ start }, stop_{ stop }
		{
		}

		GlGpuTimers* parent_{};
		GLuint64 start_{};
		GLuint64 stop_{};
	};

	auto nextFrame() -> void;

	[[nodiscard]] auto get(std::string_view label) -> float
	{
		if (results_[completedPoolIndex_].contains(label))
		{
			return results_[completedPoolIndex_][label];
		}
		return 0.0f;
	}

	[[nodiscard]] auto getAllCurrent() const -> const std::vector<b3d::profiler::ProfilerResult>&
	{
		return lastFrameResults_;
	}

	[[nodiscard]] auto record(std::string label) -> Record;

	GlGpuTimers();
	~GlGpuTimers();

private:
	friend Record;
	std::array<std::array<GLuint64, PoolSize>, DoubleBufferedFrames> eventsPool_;


	std::array<std::unordered_map<std::string, Timings>, DoubleBufferedFrames> labeledTimestamps_;
	std::array<std::unordered_map<std::string, float>, DoubleBufferedFrames> results_;
	int currentFrameIndex_{ 0 };
	int completedPoolIndex_{ 0 };
	int currentPoolIndex_{ 0 };

	int nextFreeEvent_{ 0 };
	bool waitOnReady_{ WaitOnNotReady };
	int startedRecord_{ 0 };

	bool isFirstRecord_{ true };
	std::string frameFirstRecord_;
	std::vector<b3d::profiler::ProfilerResult> lastFrameResults_{};
};
template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
auto GlGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::Record::start() const -> void
{
	++parent_->startedRecord_;
	GL_CALL(glQueryCounter(start_, GL_TIMESTAMP));
}
template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
inline auto GlGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::Record::stop() const -> void
{
	--parent_->startedRecord_;
	GL_CALL(glQueryCounter(stop_, GL_TIMESTAMP));
}
template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
auto GlGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::nextFrame() -> void
{
	assert(startedRecord_ == 0);
	currentFrameIndex_++;
	completedPoolIndex_ = (currentFrameIndex_ + 1) % DoubleBufferedFrames;
	currentPoolIndex_ = currentFrameIndex_ % DoubleBufferedFrames;

	nextFreeEvent_ = 0;
	isFirstRecord_ = true;

	labeledTimestamps_[currentPoolIndex_].clear();
	// TODO: Investigate: do we require previews result clearing??
	// results_[completedPoolIndex_].clear();
	auto availableAll = true;
	GLint available = 0;

	for (auto event : labeledTimestamps_[completedPoolIndex_])
	{
		auto start = event.second.start;
		auto stop = event.second.stop;
		/*const auto r1 = cudaEventQuery(start);
		const auto r2 = cudaEventQuery(stop);*/


		GL_CALL(glGetQueryObjectiv(stop, GL_QUERY_RESULT_AVAILABLE, &available));

		if (waitOnReady_)
		{
			while (!available)
			{
				GL_CALL(glGetQueryObjectiv(stop, GL_QUERY_RESULT_AVAILABLE, &available));
			}
		}
		availableAll = availableAll && available;
		if (available)
		{
			uint64_t timeStart;
			uint64_t timeEnd;
			GL_CALL(glGetQueryObjectui64v(start, GL_QUERY_RESULT, &timeStart));
			GL_CALL(glGetQueryObjectui64v(stop, GL_QUERY_RESULT, &timeEnd));
			float ms = (float)(timeEnd - timeStart);
			results_[completedPoolIndex_][event.first] = ms;
		}
	}

	lastFrameResults_.clear();
	lastFrameResults_.reserve(results_.size());


	if (availableAll)
	{
		uint64_t frameStart = 0;
		if (labeledTimestamps_[completedPoolIndex_].contains(frameFirstRecord_))
		{
			const auto frameStartEvent = labeledTimestamps_[completedPoolIndex_][frameFirstRecord_].start;
			GL_CALL(glGetQueryObjectui64v(frameStartEvent, GL_QUERY_RESULT, &frameStart));
		}
		for (auto& [lable, timings] : labeledTimestamps_[completedPoolIndex_])
		{
			uint64_t timeStart;
			uint64_t timeEnd;
			GL_CALL(glGetQueryObjectui64v(timings.start, GL_QUERY_RESULT, &timeStart));
			GL_CALL(glGetQueryObjectui64v(timings.stop, GL_QUERY_RESULT, &timeEnd));
			const auto deltaStart = timeStart - frameStart;
			const auto deltaEnd = timeEnd - frameStart;
			lastFrameResults_.push_back(b3d::profiler::ProfilerResult{ lable, static_cast<float>(deltaStart) / 1000000.0f, static_cast<float>(deltaEnd) / 1000000.0f });
		}
	}
}
template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
inline auto GlGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::record(std::string label) -> Record
{
	assert(nextFreeEvent_ < eventsPool_[currentPoolIndex_].size() - 1);

	auto start = eventsPool_[currentPoolIndex_][nextFreeEvent_];
	nextFreeEvent_++;
	auto stop = eventsPool_[currentPoolIndex_][nextFreeEvent_];
	nextFreeEvent_++;

	if (isFirstRecord_)
	{
		frameFirstRecord_ = label;
		isFirstRecord_ = false;
	}

	labeledTimestamps_[currentPoolIndex_][label] = Timings{ start, stop };
	return Record(this, start, stop);
}
template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
GlGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::GlGpuTimers()
{
	for (auto& pool : eventsPool_)
	{
		GL_CALL(glGenQueries(pool.size(), (GLuint*)pool.data()));
	}
}
template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
GlGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::~GlGpuTimers()
{
	assert(startedRecord_ == 0);

	for (auto& pool : eventsPool_)
	{
		for (auto& event : pool)
		{
			// OWL_CUDA_CHECK(cudaEventSynchronize(event));
		}
	}

	for (auto& pool : eventsPool_)
	{
		for (auto& event : pool)
		{
			// OWL_CUDA_CHECK(cudaEventDestroy(event));
			//GL_CALL(glDeleteQueries(1, event));
		}
	}
}
