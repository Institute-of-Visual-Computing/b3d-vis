#include "App.h"

auto main(const int argc, char** argv) -> int
{
	auto params = std::vector<Param>(argc);
	for (auto i = 0; i < argc; i++)
	{
		params[i].value = argv[i];
	}

	Application app;
	app.initialization(params);
	app.run();

	return EXIT_SUCCESS;
}
