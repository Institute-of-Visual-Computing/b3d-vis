set(MODULE_SIMPLE_NAME ServerClient)
set(MODULE_NAME "B3dTools${MODULE_SIMPLE_NAME}")

find_package(nlohmann_json CONFIG REQUIRED)
find_package(httplib CONFIG REQUIRED)

add_library(${MODULE_NAME} STATIC)

target_sources(
	${MODULE_NAME}
PRIVATE
	include/ServerClient.h
	src/ServerClient.cpp
)

target_include_directories(
	${MODULE_NAME}
PUBLIC
	include
)

find_package(stduuid CONFIG REQUIRED)

target_link_libraries(
	${MODULE_NAME}
PUBLIC
	B3D::Header::Shared::Common
	B3D::Header::Shared::Pipeline
	B3D::Header::Sofia
	B3D::Header::SofiaNanoPipeline
	B3D::Header::Nano
	B3D::Header::Fits
PRIVATE
	B3D::Tools::Project
	nlohmann_json::nlohmann_json
	httplib::httplib
	stduuid
)

target_compile_definitions(${MODULE_NAME} PRIVATE B3D_USE_NLOHMANN_JSON)

if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
	target_compile_options(${MODULE_NAME} PRIVATE /W4 /WX /external:W0 /external:anglebrackets)
endif()

target_compile_features(${MODULE_NAME} PRIVATE cxx_std_20)

add_library(B3D::Tools::${MODULE_SIMPLE_NAME} ALIAS ${MODULE_NAME})
set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "Tools")
set_property(TARGET ${MODULE_NAME} PROPERTY PROJECT_LABEL B3D::Tools::${MODULE_SIMPLE_NAME})
