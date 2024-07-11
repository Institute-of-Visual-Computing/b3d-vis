find_package(imgui CONFIG REQUIRED)

add_library(iconFontCppHeaders INTERFACE)

target_include_directories(iconFontCppHeaders INTERFACE /)

set_property(TARGET iconFontCppHeaders PROPERTY FOLDER "ThirdParty/iconFontCppHeaders")
