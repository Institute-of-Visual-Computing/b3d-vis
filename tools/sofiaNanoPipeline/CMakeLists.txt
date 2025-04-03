set(MODULE_SIMPLE_NAME SofiaNanoPipeline)
set(MODULE_NAME "B3dTools${MODULE_SIMPLE_NAME}")

add_library(${MODULE_NAME} STATIC)

target_sources(
	${MODULE_NAME}
PRIVATE
	include/SofiaNanoPipeline.h
	src/SofiaNanoPipeline.cpp
 )

target_include_directories(
	${MODULE_NAME}
PUBLIC
	include
)

find_package(OpenVDB CONFIG REQUIRED COMPONENTS nanovdb openvdb)

target_link_libraries(
	${MODULE_NAME}
PUBLIC
	B3D::Header::SofiaNanoPipeline
	B3D::Header::Shared::Common
	B3D::Header::Sofia
	B3D::Header::Nano
	B3D::Header::Shared::Pipeline
PRIVATE
	B3D::Tools::Sofia
	B3D::Tools::Nano
	OpenVDB::nanovdb
	OpenVDB::openvdb
)

if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
	target_compile_options(${MODULE_NAME} PRIVATE /W4 /WX /external:W0 /external:anglebrackets)
endif()

target_compile_features(${MODULE_NAME} PRIVATE cxx_std_20)

add_library(B3D::Tools::${MODULE_SIMPLE_NAME} ALIAS ${MODULE_NAME})
set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "Tools")
set_property(TARGET ${MODULE_NAME} PROPERTY PROJECT_LABEL B3D::Tools::${MODULE_SIMPLE_NAME})

# Header
####################################################################################################

set(HEADER_SIMPLE_NAME SofiaNanoPipeline)
set(HEADER_MODULE_NAME "B3DHeader${HEADER_SIMPLE_NAME}")

add_library(${HEADER_MODULE_NAME} INTERFACE)

target_include_directories(
	${HEADER_MODULE_NAME}
INTERFACE
	shared
)

add_library(B3D::Header::${HEADER_SIMPLE_NAME} ALIAS ${HEADER_MODULE_NAME})

########################################

set(MODULE_SIMPLE_CLI_NAME SofiaNanoPipelineCLI)
set(MODULE_CLI_NAME "B3dTools${MODULE_SIMPLE_CLI_NAME}")

find_package(stduuid CONFIG REQUIRED)
find_package(args CONFIG REQUIRED)
find_package(plog CONFIG REQUIRED)

add_executable(${MODULE_CLI_NAME})

target_sources(
	${MODULE_CLI_NAME}
PRIVATE
	src/SofiaNanoPipelineCLI.cpp
	src/SofiaNanoPipeline.cpp
)

target_include_directories(
	${MODULE_CLI_NAME}
PUBLIC
	include
)

target_compile_definitions(${MODULE_CLI_NAME} PRIVATE B3D_USE_NLOHMANN_JSON)

target_link_libraries(
	${MODULE_CLI_NAME}
PRIVATE
	B3D::Tools::Sofia
	B3D::Tools::Fits
	B3D::Tools::Nano
	B3D::Tools::SofiaNanoPipeline
	B3D::Tools::Project
	OpenVDB::nanovdb
	OpenVDB::openvdb
	stduuid
	taywee::args
	plog::plog
)

if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(${MODULE_CLI_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
	target_compile_options(${MODULE_CLI_NAME} PRIVATE /W4 /WX /external:W0 /external:anglebrackets)
endif()

target_compile_features(${MODULE_CLI_NAME} PRIVATE cxx_std_20)

add_executable(B3D::Tools::${MODULE_SIMPLE_CLI_NAME} ALIAS ${MODULE_CLI_NAME})
set_property(TARGET ${MODULE_CLI_NAME} PROPERTY FOLDER "Tools")
set_property(TARGET ${MODULE_CLI_NAME} PROPERTY PROJECT_LABEL B3D::Tools::${MODULE_SIMPLE_CLI_NAME})
