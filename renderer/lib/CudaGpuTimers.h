#pragma once

#include <array>
#include <string>
#include <unordered_map>

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <owl/helper/cuda.h>


namespace b3d::renderer
{
	template <int PoolSize, int DoubleBufferedFrames = 2, bool WaitOnNotReady = false>
	class CudaGpuTimers
	{
	public:
		class Record
		{
		public:
			auto start() const -> void;
			auto stop() const -> void;

		private:
			friend CudaGpuTimers;

			Record(CudaGpuTimers* parent, const CUstream stream, const cudaEvent_t start, const cudaEvent_t stop)
				: parent_{ parent }, stream_{ stream }, start_{ start }, stop_{ stop }
			{
			}

			CudaGpuTimers* parent_{};
			CUstream stream_{};
			cudaEvent_t start_{};
			cudaEvent_t stop_{};
		};

		auto nextFrame() -> void;
		[[nodiscard]] auto get(const std::string& label) -> float
		{
			if(results_[completedPoolIndex_].contains(label))
			{
				return results_[completedPoolIndex_][label];
			}
			return 0.0f;
		}

		[[nodiscard]] auto record(std::string label, CUstream stream) -> Record;

		CudaGpuTimers();
		virtual ~CudaGpuTimers();

	private:
		friend Record;
		std::array<std::array<cudaEvent_t, PoolSize>, DoubleBufferedFrames> eventsPool_;
		std::array<std::unordered_map<std::string, std::pair<cudaEvent_t, cudaEvent_t>>, DoubleBufferedFrames> labeledTimestamps_;
		std::array<std::unordered_map<std::string, float>, DoubleBufferedFrames> results_;
		int currentFrameIndex_{ 0 };
		int completedPoolIndex_{ 0 };
		int currentPoolIndex_{ 0 };

		int nextFreeEvent_{ 0 };
		bool waitOnReady_{ WaitOnNotReady };
		int startedRecord_{ 0 };
	};
	template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
	auto CudaGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::Record::start() const -> void
	{
		++parent_->startedRecord_;
		cudaEventRecord(start_, stream_);
	}
	template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
	inline auto CudaGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::Record::stop() const -> void
	{
		--parent_->startedRecord_;
		cudaEventRecord(stop_, stream_);
	}
	template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
	auto CudaGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::nextFrame() -> void
	{
		assert(startedRecord_ == 0);
		currentFrameIndex_++;
		completedPoolIndex_ = (currentFrameIndex_ + 1) % DoubleBufferedFrames;
		currentPoolIndex_ = currentFrameIndex_ % DoubleBufferedFrames;

		nextFreeEvent_ = 0;

		results_[completedPoolIndex_].clear();

		for (auto event : labeledTimestamps_[completedPoolIndex_])
		{
			auto start = event.second.first;
			auto stop = event.second.second;
			const auto r1 = cudaEventQuery(start);
			const auto r2 = cudaEventQuery(stop);

			if (waitOnReady_)
			{
				if (r2 == cudaErrorNotReady)
				{
					OWL_CUDA_CHECK(cudaEventSynchronize(stop));
				}
			}

			if (r1 == cudaSuccess && r2 == cudaSuccess)
			{
				float ms;
				OWL_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
				results_[completedPoolIndex_][event.first] = ms;
			}

			auto invalid = r1 != cudaSuccess && r1 != cudaErrorNotReady && r2 != cudaSuccess && r2 != cudaErrorNotReady;
			assert(!invalid);
		}
		labeledTimestamps_[completedPoolIndex_].clear();
	}
	template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
	inline auto CudaGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::record(std::string label,
																					  CUstream stream) -> Record
	{
		assert(nextFreeEvent_ < eventsPool_[currentPoolIndex_].size() - 1);
		auto start = eventsPool_[currentPoolIndex_][nextFreeEvent_];
		nextFreeEvent_++;
		auto stop = eventsPool_[currentPoolIndex_][nextFreeEvent_];
		nextFreeEvent_++;

		labeledTimestamps_[currentPoolIndex_][label] = std::make_pair(start, stop);

		return Record(this, stream, start, stop);
	}
	template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
	CudaGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::CudaGpuTimers()
	{
		for (auto& pool : eventsPool_)
		{
			for (auto& event : pool)
			{
				OWL_CUDA_CHECK(cudaEventCreate(&event));
			}
		}
	}
	template <int PoolSize, int DoubleBufferedFrames, bool WaitOnNotReady>
	CudaGpuTimers<PoolSize, DoubleBufferedFrames, WaitOnNotReady>::~CudaGpuTimers()
	{
		assert(startedRecord_ == 0);

		for (auto& pool : eventsPool_)
		{
			for (auto& event : pool)
			{
				OWL_CUDA_CHECK(cudaEventSynchronize(event));
			}
		}

		for (auto& pool : eventsPool_)
		{
			for (auto& event : pool)
			{
				OWL_CUDA_CHECK(cudaEventDestroy(event));
			}
		}
	}
} // namespace b3d::renderer
