set(HEADER_SIMPLE_NAME Pipeline)
set(HEADER_MODULE_NAME "B3DHeader${HEADER_SIMPLE_NAME}")

add_library(${HEADER_MODULE_NAME} INTERFACE)

target_include_directories(
	${HEADER_MODULE_NAME}
INTERFACE
	./
)

add_library(B3D::Header::Shared::${HEADER_SIMPLE_NAME} ALIAS ${HEADER_MODULE_NAME})

# Header JSON
####################################################################################################
find_package(nlohmann_json CONFIG REQUIRED)

set(HEADER_SIMPLE_NAME Pipeline)
set(HEADER_MODULE_NAME "B3DHeader${HEADER_SIMPLE_NAME}Json")

add_library(${HEADER_MODULE_NAME} INTERFACE)

target_include_directories(
	${HEADER_MODULE_NAME}
INTERFACE
	./
)

target_compile_definitions(${HEADER_MODULE_NAME} INTERFACE B3D_USE_NLOHMANN_JSON)
add_library(B3D::Header::${HEADER_SIMPLE_NAME}::Json ALIAS ${HEADER_MODULE_NAME})
