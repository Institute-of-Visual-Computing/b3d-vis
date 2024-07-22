find_package(glfw3 CONFIG REQUIRED)
set(owl_dir ${CMAKE_CURRENT_LIST_DIR}/owl)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
add_subdirectory(${owl_dir}/samples/common)
# add_subdirectory(${owl_dir}/samples/common)

add_subdirectory(imGuizmo)
add_subdirectory(imCurve)

set_property(TARGET owl PROPERTY FOLDER "ThirdParty/owl")
set_property(TARGET owl_viewer PROPERTY FOLDER "ThirdParty/owl")

add_subdirectory(iconFontCppHeaders)
