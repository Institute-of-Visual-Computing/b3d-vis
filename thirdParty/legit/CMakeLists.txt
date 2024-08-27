add_library(legit INTERFACE ../ImGuiUtils/ImGuiProfilerRenderer.h ../ImGuiUtils/ProfilerTask.h)
target_include_directories(legit INTERFACE ../ImGuiUtils/)
set_property(TARGET legit PROPERTY FOLDER "ThirdParty/legit")
