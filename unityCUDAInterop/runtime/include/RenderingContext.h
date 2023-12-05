#pragma once
namespace b3d::unity_cuda_interop
{
	class SyncPrimitive;
}
namespace b3d::unity_cuda_interop
{
	class RenderingContext
	{
	public:
		virtual ~RenderingContext()
		{
		}

		virtual auto signal(SyncPrimitive* syncPrimitive, unsigned value) -> void = 0;
		virtual auto wait(SyncPrimitive* syncPrimitive, unsigned value) -> void = 0; 

		auto isValid() const -> bool
		{
			return isValid_;
		}

	protected:
		RenderingContext()
		{
		}

		bool isValid_{ false };
	};
} // namespace b3d::unity_cuda_interop
