
find_package(nlohmann_json CONFIG REQUIRED)
add_library(NanoCutterParser)
target_sources(NanoCutterParser PUBLIC FILE_SET HEADERS FILES NanoCutterParser.h)
target_sources(NanoCutterParser PRIVATE NanoCutterParser.cpp)
target_link_libraries(NanoCutterParser PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(NanoCutterParser PUBLIC cxx_std_20)
set_target_properties(NanoCutterParser PROPERTIES LINKER_LANGUAGE CXX)
if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(NanoCutterParser PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
endif()


set_property(TARGET NanoCutterParser PROPERTY FOLDER "Tools")
