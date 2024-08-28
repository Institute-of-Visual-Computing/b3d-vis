add_library(legit_task INTERFACE ProfilerTask.h)
target_include_directories(legit_task INTERFACE /)
set_property(TARGET legit_task PROPERTY FOLDER "ThirdParty/legit_task")

add_library(legit INTERFACE ImGuiProfilerRenderer.h)
target_include_directories(legit INTERFACE /)
target_link_libraries(legit
INTERFACE
	legit_task
)
set_property(TARGET legit PROPERTY FOLDER "ThirdParty/legit")
