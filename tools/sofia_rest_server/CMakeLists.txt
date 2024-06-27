find_package(httplib CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
 find_package(args CONFIG REQUIRED)
add_executable(sofia_server main.cpp)

target_link_libraries(sofia_server
	PRIVATE
		httplib::httplib
		nlohmann_json::nlohmann_json
		Boost::boost
		Boost::filesystem
		taywee::args
)

target_compile_features(sofia_server PUBLIC cxx_std_20)

if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(NanoCutterParser PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
endif()

