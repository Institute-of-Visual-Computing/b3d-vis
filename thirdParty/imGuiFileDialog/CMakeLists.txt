include(FetchContent)

FetchContent_Declare(
  ImGuiFileDialog
  GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
  GIT_TAG        master
)


FetchContent_MakeAvailable(ImGuiFileDialog)


find_package(imgui CONFIG REQUIRED)

add_library(imGuiFileDialog STATIC)
target_sources(imGuiFileDialog PUBLIC
	${ImGuiFileDialog_SOURCE_DIR}/ImGuiFileDialog.cpp
	${ImGuiFileDialog_SOURCE_DIR}/ImGuiFileDialog.h
	CustomImGuiFileDialogConfig.h
)

target_include_directories(imGuiFileDialog PUBLIC ${ImGuiFileDialog_SOURCE_DIR})

target_compile_features(imGuiFileDialog PUBLIC cxx_std_23)
target_link_libraries(imGuiFileDialog 
PRIVATE
	imgui::imgui
	iconFontCppHeaders
)

target_compile_definitions(imGuiFileDialog PUBLIC CUSTOM_IMGUIFILEDIALOG_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/CustomImGuiFileDialogConfig.h")

set_property(TARGET imGuiFileDialog PROPERTY FOLDER "ThirdParty/imGuiFileDialog")


set_target_properties(ImGuiFileDialog PROPERTIES FOLDER "thirdParty/imGuiFileDialog_unused")
