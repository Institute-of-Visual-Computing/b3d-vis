find_package(nlohmann_json CONFIG REQUIRED)

set(MODULE_NAME "ColorMap")

add_library(${MODULE_NAME} STATIC)

target_sources(
	${MODULE_NAME}
PRIVATE
	include/ColorMap.h
	src/ColorMap.cpp
)

target_include_directories(
	${MODULE_NAME}
PUBLIC
	include
)

target_link_libraries(${MODULE_NAME} PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(${MODULE_NAME} PRIVATE cxx_std_20)
set_target_properties(${MODULE_NAME} PROPERTIES LINKER_LANGUAGE CXX)
if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
	target_compile_options(${MODULE_NAME} PRIVATE /W4 /WX /external:W0 /external:anglebrackets)
endif()

add_library(B3D::Tools::${MODULE_NAME} ALIAS ${MODULE_NAME})
set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "Tools")
