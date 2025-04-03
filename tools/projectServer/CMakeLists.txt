find_package(httplib CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(args CONFIG REQUIRED)
find_package(stduuid CONFIG REQUIRED)
find_package(plog CONFIG REQUIRED)

set(MODULE_NAME "ProjectServer")

add_executable(${MODULE_NAME})

target_sources(
	${MODULE_NAME}
PRIVATE
	main.cpp
	src/ProjectProvider.cpp
	src/ProjectProvider.h
	src/Internals.h
)

target_link_libraries(
	${MODULE_NAME}
PRIVATE
	B3D::Tools::Project
	B3D::Tools::FileCatalog
	B3D::Tools::Sofia
	B3D::Tools::Nano
	B3D::Tools::Fits
	B3D::Tools::SofiaNanoPipeline
	B3D::Header::Shared::Helper
	nlohmann_json::nlohmann_json
	httplib::httplib
	Boost::boost
	Boost::filesystem
	taywee::args
	stduuid
	plog::plog
)

target_compile_definitions(${MODULE_NAME} PUBLIC B3D_USE_NLOHMANN_JSON)
target_compile_features(${MODULE_NAME} PRIVATE cxx_std_20)

if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
	target_compile_options(${MODULE_NAME} PRIVATE /W4 /WX /external:W0 /external:anglebrackets)
endif()
