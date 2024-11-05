set(MODULE_SIMPLE_NAME SofiaNanoPipeline)
set(MODULE_NAME "B3dTools${MODULE_SIMPLE_NAME}")

add_library(${MODULE_NAME})

target_sources(${MODULE_NAME}
	PUBLIC
		include/SofiaNanoPipeline.h
	PRIVATE
		src/SofiaNanoPipeline.cpp
		src/include/SofiaNanoPipeline_Internal.h
 )

target_include_directories(${MODULE_NAME}
	PUBLIC
		"include"
	PRIVATE
		"src"
)

target_link_libraries(${MODULE_NAME}
	PUBLIC
		B3D::Tools::Sofia
		B3D::Tools::Nano
		B3D::Header::SofiaNanoPipeline
		B3D::Header::Shared::Common
)

if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
endif()

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_20 PRIVATE cxx_std_20)

add_library(B3D::Tools::${MODULE_SIMPLE_NAME} ALIAS ${MODULE_NAME})
set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "Tools")
# Currently ignored by VS
set_property(TARGET ${MODULE_NAME} PROPERTY PROJECT_LABEL B3D::Tools::${MODULE_SIMPLE_NAME})

# Header
####################################################################################################

set(HEADER_SIMPLE_NAME SofiaNanoPipeline)
set(HEADER_MODULE_NAME "B3DHeader${HEADER_SIMPLE_NAME}")

add_library(${HEADER_MODULE_NAME} INTERFACE)

target_sources(${HEADER_MODULE_NAME}
	INTERFACE
		shared/PipelineResult.h
 )

target_include_directories(${HEADER_MODULE_NAME}
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

add_executable(${MODULE_CLI_NAME} src/SofiaNanoPipelineCLI.cpp)

target_sources(${MODULE_CLI_NAME}
	PUBLIC
		src/SofiaNanoPipelineCLI.cpp
)

target_include_directories(${MODULE_CLI_NAME}
	PUBLIC
		include
		src
)

target_compile_definitions(${MODULE_CLI_NAME} PRIVATE B3D_USE_NLOHMANN_JSON)

target_link_libraries(${MODULE_CLI_NAME}
	PUBLIC
		B3D::Tools::Sofia
		B3D::Tools::Fits
		B3D::Tools::Nano
		B3D::Tools::SofiaNanoPipeline
		B3D::Tools::Project
		B3D::Header::Shared::Helper
		B3D::Header::Shared::Pipeline
		stduuid
		taywee::args
		plog::plog
)

if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(${MODULE_CLI_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
endif()


target_compile_features(${MODULE_CLI_NAME} PUBLIC cxx_std_20 PRIVATE cxx_std_20)

add_executable(B3D::Tools::${MODULE_SIMPLE_CLI_NAME} ALIAS ${MODULE_CLI_NAME})
set_property(TARGET ${MODULE_CLI_NAME} PROPERTY FOLDER "Tools")
# Currently ignored by VS
set_property(TARGET ${MODULE_CLI_NAME} PROPERTY PROJECT_LABEL B3D::Tools::${MODULE_SIMPLE_CLI_NAME})
