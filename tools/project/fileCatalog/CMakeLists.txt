set(MODULE_SIMPLE_NAME FileCatalog)
set(MODULE_NAME "B3dTools${MODULE_SIMPLE_NAME}")

find_package(stduuid CONFIG REQUIRED)


add_library(${MODULE_NAME})

target_sources(${MODULE_NAME}
	PUBLIC
		include/FileCatalog.h
	PRIVATE
		src/FileCatalog.cpp
 )

target_include_directories(${MODULE_NAME}
	PUBLIC
		"include"
	PRIVATE
		"src"
)

target_link_libraries(${MODULE_NAME}
	PUBLIC
	PRIVATE
		stduuid
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
