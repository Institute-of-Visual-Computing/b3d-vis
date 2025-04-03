set(MODULE_SIMPLE_NAME Fits)
set(MODULE_NAME "B3dTools${MODULE_SIMPLE_NAME}")

find_package(unofficial-cfitsio CONFIG REQUIRED)

add_library(${MODULE_NAME} STATIC)

target_sources(
	${MODULE_NAME}	
PRIVATE
	include/FitsTools.h
	src/FitsTools.cpp
 )

target_include_directories(
	${MODULE_NAME}
PUBLIC
	include
)

target_link_libraries(
	${MODULE_NAME}
PUBLIC
	B3D::Header::Shared::Common
	B3D::Header::Fits
PRIVATE
	cfitsio
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

set(HEADER_SIMPLE_NAME Fits)
set(HEADER_MODULE_NAME "B3DHeader${HEADER_SIMPLE_NAME}")

add_library(${HEADER_MODULE_NAME} INTERFACE)
target_sources(
	${HEADER_MODULE_NAME}
INTERFACE
	shared/FitsCommon.h
 )
target_include_directories(
	${HEADER_MODULE_NAME}
INTERFACE
	shared
)
add_library(B3D::Header::${HEADER_SIMPLE_NAME} ALIAS ${HEADER_MODULE_NAME})

# Header JSON
####################################################################################################
find_package(nlohmann_json CONFIG REQUIRED)

set(HEADER_SIMPLE_NAME Fits)
set(HEADER_MODULE_NAME "B3DHeader${HEADER_SIMPLE_NAME}Json")

add_library(${HEADER_MODULE_NAME} INTERFACE)
target_sources(
	${HEADER_MODULE_NAME}
INTERFACE
	shared/FitsCommon.h
 )
target_include_directories(
	${HEADER_MODULE_NAME}
INTERFACE
	shared
)

target_compile_definitions(${HEADER_MODULE_NAME} INTERFACE B3D_USE_NLOHMANN_JSON)
add_library(B3D::Header::${HEADER_SIMPLE_NAME}::Json ALIAS ${HEADER_MODULE_NAME})
