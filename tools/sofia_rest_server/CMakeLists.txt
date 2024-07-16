find_package(httplib CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(args CONFIG REQUIRED)
find_package(stduuid CONFIG REQUIRED)

add_executable(sofia_server main.cpp "src/SoFiA.h" "src/SoFiA.cpp" "src/ProjectProvider.cpp" "src/ProjectProvider.h")

target_link_libraries(sofia_server
	PRIVATE
		httplib::httplib
		nlohmann_json::nlohmann_json
		Boost::boost
		Boost::filesystem
		taywee::args
		B3D::Tools::ProjectExplorer
		stduuid
		NanoCutApi
)

target_include_directories(sofia_server
	PRIVATE
		src

)


target_compile_features(sofia_server PUBLIC cxx_std_20 PRIVATE cxx_std_20)

if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(NanoCutterParser PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
endif()

