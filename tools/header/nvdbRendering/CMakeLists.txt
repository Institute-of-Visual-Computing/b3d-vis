set(HEADER_SIMPLE_NAME NvdbRendering)
set(HEADER_MODULE_NAME "B3DHeader${HEADER_SIMPLE_NAME}")

add_library(${HEADER_MODULE_NAME} INTERFACE)

target_include_directories(
	${HEADER_MODULE_NAME}
INTERFACE
	./
)

add_library(B3D::Header::Shared::${HEADER_SIMPLE_NAME} ALIAS ${HEADER_MODULE_NAME})

