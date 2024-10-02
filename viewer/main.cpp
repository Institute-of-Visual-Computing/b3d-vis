

#ifdef NANO_PROFILE
#include <tracy/Tracy.hpp>
thread_local auto currentStackDepth = 60;

auto operator new(std::size_t count) -> void*
{
	auto ptr = malloc(count);
	TracyAllocS(ptr, count, currentStackDepth);
	return ptr;
}
auto operator delete(void* ptr) noexcept -> void
{
	TracyFreeS(ptr, currentStackDepth);
	free(ptr);
}
#endif


#include "App.h"

auto main(const int argc, char** argv) -> int
{
	/*_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);*/

	auto params = std::vector<Param>(argc);
	for (auto i = 0; i < argc; i++)
	{
		params[i].value = argv[i];
	}

	Application app;
	app.initialization(params);
	app.run();
	

	/*_CrtDumpMemoryLeaks();*/

	return EXIT_SUCCESS;
}
