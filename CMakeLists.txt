cmake_minimum_required(VERSION 3.26)

set(b3d_version_major 0)
set(b3d_version_minor 1)
set(b3d_version_patch 0)

project(b3d_visualisation
    VERSION ${b3d_version_major}.${b3d_version_minor}.${b3d_version_patch}
    LANGUAGES CXX C CUDA
)

option( BUILD_SERVER "Build server component" ON )
option( BUILD_UNITY_EXTENSION "Build renderer as an Unity extention" OFF )
option( BUILD_DESKTOP_CLIENT_VIEWER "Build desktop viewer" ON )

option( ENABLE_NANO_VIEWER_PROFILE "Enables Tracy profiling for a desktop viewer" OFF )
option( ENABLE_NANO_VIEWER_SANITIZERS "Enables address sanitizers" OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

function(add_b3d_renderer)

	set(options)
	set(oneValueArgs NAME)
	set(multiValueArgs SOURCES HEADERS LIBRARIES SHARED_LIBRARIES)
	cmake_parse_arguments(TARGET "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

	if(NOT TARGET_NAME)
		message(FATAL_ERROR "NAME must be defined")
	endif()

	#TODO: Move static/dynamic to optianal parameter
	add_library(${TARGET_NAME} STATIC)
	target_sources(${TARGET_NAME} PRIVATE ${TARGET_SOURCES} ${TARGET_HEADERS})
	target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
	if(MSVC)
		set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
		target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
		target_compile_options(${TARGET_NAME} PRIVATE /W4 /WX /external:W0 /external:anglebrackets)
	endif()
	target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

	target_link_libraries(${TARGET_NAME}
		PRIVATE
			B3D::Renderer::Framework
			"${TARGET_LIBRARIES}"
	)
	set_target_properties(${TARGET_NAME}
		PROPERTIES
			CUDA_STANDARD 20
			FOLDER "Renderer"
	)

	add_library(B3D::Renderer::${TARGET_NAME} ALIAS ${TARGET_NAME})
endfunction()

add_subdirectory(thirdParty)
add_subdirectory(renderer)
add_subdirectory(tools)

if(BUILD_DESKTOP_CLIENT_VIEWER)
	add_subdirectory(viewer)
endif()
if(BUILD_UNITY_EXTENSION)
	add_subdirectory(unityCUDAInterop)
endif()
