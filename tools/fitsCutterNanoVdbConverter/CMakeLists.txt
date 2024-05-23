
find_package(unofficial-cfitsio CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenVDB CONFIG REQUIRED COMPONENTS openvdb nanovdb)

add_executable(NanoCutter)
target_sources(NanoCutter PRIVATE
	main.cpp
	FitsHelper.h
	Common.h
	FitsHelper.cpp
	NanoGenerator.cpp
	ClusterProcessor.h
	BinaryPartitionClusterProcessor.h
)

target_link_libraries(NanoCutter PRIVATE cfitsio Boost::program_options OpenVDB::openvdb OpenVDB::nanovdb NanoCutterParser)
target_compile_features(NanoCutter PUBLIC cxx_std_20)
if(MSVC)
	set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
	target_compile_options(NanoCutter PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
endif()


set_property(TARGET NanoCutter PROPERTY FOLDER "Tools")




add_library(NanoCutApi SHARED)
target_sources(NanoCutApi PRIVATE
	NanoCutApi.cpp
	Common.h
	FitsHelper.cpp
	NanoGenerator.cpp
)
target_link_libraries(NanoCutApi PUBLIC cfitsio OpenVDB::openvdb OpenVDB::nanovdb)
target_compile_features(NanoCutApi PUBLIC cxx_std_20)
target_include_directories(NanoCutApi PUBLIC /include)

if(MSVC)
	target_compile_options(NanoCutApi PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
endif()


#find_package(Catch2 REQUIRED)
#add_executable(NanoCutApi_Tests)
#target_sources(NanoCutApi_Tests
#PRIVATE
#	NanoCutApi_Tests.cpp
#	FitsHelper.cpp
#	NanoGenerator.cpp
#)

#target_link_libraries(NanoCutApi_Tests PUBLIC NanoCutApi Catch2::Catch2 Catch2::Catch2WithMain cfitsio OpenVDB::openvdb OpenVDB::nanovdb)
#target_compile_features(NanoCutApi_Tests PUBLIC cxx_std_20)
#set_property(TARGET NanoCutApi_Tests PROPERTY
#             MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

#include(CTest)
#include(Catch)
#catch_discover_tests(NanoCutApi_Tests)
