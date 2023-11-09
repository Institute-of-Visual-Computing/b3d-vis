#pragma once

#include <memory>
#include <string>
#include <vector>
#include <owl/common.h>

namespace b3d
{


	struct Camera
	{
		owl::vec3f origin;
		owl::vec3f at;
		owl::vec3f up;
		float cosFoV;
	};
	struct View
	{
		Camera camera1;
	};

	class RendererBase
	{
	  public:
		inline void initialize()
		{
			onInitialize();
		}
		inline void deinitialize()
		{
			onDeinitialize();
		}

		inline void gui()
		{
			onGui();
		}

		inline void render(const View& view)
		{
			onRender(view);
		}

	  protected:
		virtual void onInitialize(){};
		virtual void onDeinitialize(){};

		virtual void onGui(){};
		virtual void onRender(const View& view) = 0;
	};

	void addRenderer(std::shared_ptr<RendererBase> renderer, const std::string& name);

	struct RendererRegistryEntry
	{
		std::shared_ptr<RendererBase> rendererInstance;
		std::string name;
	};

	extern std::vector<RendererRegistryEntry> registry;

	template <typename T> void registerRenderer(const std::string& name)
	{
		registry.push_back({std::make_shared<T>(), name});
	}

} // namespace b3d
