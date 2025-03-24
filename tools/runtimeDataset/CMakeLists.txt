set(MODULE_SIMPLE_NAME RuntimeDataset)
set(MODULE_NAME "B3dTools${MODULE_SIMPLE_NAME}")

find_package(OpenVDB CONFIG REQUIRED COMPONENTS nanovdb)

add_library(${MODULE_NAME})

target_sources(${MODULE_NAME}
	PUBLIC
		include/RuntimeDataset.h
	PRIVATE
		src/RuntimeDataset.cpp
		src/include/RuntimeDataset_Internal.h
)

target_include_directories(${MODULE_NAME}
	PUBLIC
		include
	PRIVATE
		src
)

target_link_libraries(${MODULE_NAME}
	PUBLIC
		OpenVDB::nanovdb
		B3D::Header::Shared::Common
		B3D::Header::Shared::NvdbRendering
	PRIVATE
		owl::owl
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
