cmake_minimum_required(VERSION 3.26)

project(b3d_visualisation
    VERSION 0.1
    LANGUAGES CXX C CUDA
)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#it a hack, otherwise it will be included by owl package with wrong settings
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit)

#TODO: move owl stuff to its own cmakelists file
set(owl_dir ${PROJECT_SOURCE_DIR}/thirdParty/owl)
OPTION(OWL_BUILD_SAMPLES OFF)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
add_subdirectory(${owl_dir}/samples/common)


function(add_b3d_renderer)

	set(options)
	set(oneValueArgs NAME)
	set(multiValueArgs SOURCES HEADERS LIBRARIES)
	cmake_parse_arguments(TARGET "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

	if(NOT TARGET_NAME)
		message(FATAL_ERROR "NAME must be defined")
	endif()

	set(MODULE_NAME ${TARGET_NAME})
	set(SRC ${TARGET_SOURCES})

	add_library(${MODULE_NAME} ${SRC})
	target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)
	target_include_directories(${MODULE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

	target_link_libraries(${MODULE_NAME}
		PRIVATE
			B3D::Renderer::Library
			"${TARGET_LIBRARIES}"
	)
	set_property(TARGET ${MODULE_NAME} PROPERTY CUDA_STANDARD 17)



	add_library(B3D::Renderer::${MODULE_NAME} ALIAS ${MODULE_NAME})


	set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "Renderer")

endfunction()


add_subdirectory(renderer)
add_subdirectory(viewer)
add_subdirectory(unityCUDAInterop)


