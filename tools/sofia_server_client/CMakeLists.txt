set(MODULE_NAME SofiaServerClient)


find_package(httplib CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)


add_library(${MODULE_NAME}
	"src/Client.cpp"
	"include/Client.h"
)

target_link_libraries(${MODULE_NAME}
	PRIVATE
		httplib::httplib
		nlohmann_json::nlohmann_json
	PUBLIC
		B3D::Tools::ProjectExplorer
)

target_include_directories(${MODULE_NAME}
	PRIVATE
		"src"
	PUBLIC
		"include"
)

if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
endif()


target_compile_features(${MODULE_NAME} PUBLIC cxx_std_20 PRIVATE cxx_std_20)

add_library(B3D::Tools::${MODULE_NAME} ALIAS ${MODULE_NAME})
set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "Tools")
