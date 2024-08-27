add_library(legit INTERFACE ImGuiProfilerRenderer.h ProfilerTask.h)
target_include_directories(legit INTERFACE /)
set_property(TARGET legit PROPERTY FOLDER "ThirdParty/legit")
