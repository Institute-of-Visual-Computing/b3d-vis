find_package(glfw3 CONFIG REQUIRED)
set(owl_dir ${CMAKE_CURRENT_LIST_DIR}/owl)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

add_subdirectory(imGuizmo)
add_subdirectory(imCurve)
add_subdirectory(imSpinner)
add_subdirectory(legit)

set_property(TARGET owl PROPERTY FOLDER "ThirdParty/owl")

add_subdirectory(iconFontCppHeaders)
