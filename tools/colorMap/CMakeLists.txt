find_package(nlohmann_json CONFIG REQUIRED)

add_library(ColorMap
	src/ColorMap.cpp
)

target_include_directories(ColorMap
	PUBLIC
		include
)

target_link_libraries(ColorMap PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(ColorMap PUBLIC cxx_std_20)
set_target_properties(ColorMap PROPERTIES LINKER_LANGUAGE CXX)
if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(ColorMap PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
endif()

add_library(B3D::Tools::ColorMap ALIAS ColorMap)
set_property(TARGET ColorMap PROPERTY FOLDER "Tools")
