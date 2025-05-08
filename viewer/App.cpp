#include "App.h"

#include <functional>

#include "NanoViewer.h"


#include <FitsNvdbRenderer.h>
#include <NullRenderer.h>


using namespace b3d::renderer;

namespace
{
	auto rendererIndex = 0;
	auto disableVsync = false;
	auto shouldRun = true;
	auto enableDevMode = false;

	struct Command
	{
		std::string name;
		std::string description;
	};

	std::vector<Command> commands;

	auto showHelp() -> void
	{
		for (const auto& [name, description] : commands)
		{
			std::cout << "\t--" << name << "\t\t" << description << std::endl;
		}
	}

	auto notifyFaultyArguments() -> void
	{
		showHelp();
		shouldRun = false;
	}


	auto addParamCommand(const std::vector<Param>& vector, const std::string& commandName,
						 const std::string& description,
						 const std::function<void(const std::vector<std::string>&)>& callback) -> void
	{

		commands.push_back(Command{ commandName, description });

		auto foundBegin = std::ranges::find_if(vector,
											   [&](const Param& param)
											   {
												   if (param.value == "--" + commandName)
												   {
													   return true;
												   }
												   return false;
											   });
		if (foundBegin != vector.end())
		{
			auto foundEnd = std::find_if(foundBegin + 1, vector.end(),
										 [&](const Param& param)
										 {
											 if (param.value.substr(0, 2) == "--")
											 {
												 return true;
											 }
											 return false;
										 });


			auto values = std::vector<std::string>{};

			for (++foundBegin; foundBegin != foundEnd; ++foundBegin)
			{
				const auto [value] = *foundBegin;
				values.push_back(value);
			}

			callback(values);
		}
	}
} // namespace

auto Application::run() -> void
{
	if (!shouldRun)
	{
		return;
	}
	std::cout << registry.front().name << std::endl;
	using namespace std::string_literals;
	auto viewer = NanoViewer{ "B3D HI-Datacube Inspector"s, 1980, 1080, !disableVsync, rendererIndex };

	viewer.enableDevelopmentMode(enableDevMode);
	viewer.run();
}

auto Application::initialization(const std::vector<Param>& parameters) -> void
{
	registerRenderer<FitsNvdbRenderer>("FitsNvdbRenderer");
	registerRenderer<NullRenderer>("nullRenderer");

	addParamCommand(parameters, "renderer", "Sets default renderer.",
					[&](const std::vector<std::string>& values)
					{
						if (values.size() == 1)
						{
							const auto value = values.front();
							const auto index = getRendererIndex(value);

							if (index != -1)
							{
								rendererIndex = index;
							}
						}
						else
						{
							notifyFaultyArguments();
						}
					});

	addParamCommand(parameters, "disable_vsync", "Disables VSync.",
					[&](const std::vector<std::string>& values)
					{
						if (values.size() == 0)
						{
							disableVsync = true;
						}
						else
						{
							notifyFaultyArguments();
						}
					});
	addParamCommand(parameters, "enable_vsync", "Enables VSync.",
					[&](const std::vector<std::string>& values)
					{
						if (values.size() == 0)
						{
							disableVsync = false;
						}
						else
						{
							notifyFaultyArguments();
						}
					});
	addParamCommand(parameters, "enable_dev_mode", "Enables under development features.",
					[&](const std::vector<std::string>& values)
					{
						if (values.size() == 0)
						{
							enableDevMode = true;
						}
						else
						{
							notifyFaultyArguments();
						}
					});
	addParamCommand(parameters, "help", "Show help.",
					[&](const std::vector<std::string>& values)
					{
						if (values.size() == 0)
						{
							shouldRun = false;
							showHelp();
						}
						else
						{
							notifyFaultyArguments();
						}
					});
}
