set(UNITY_ASSET_SUBDIRECTORY_INTEROP_ACTIONS "UnityCudaInterop/Actions")

if(not defined UNITY_PROJECT_ROOT_DIR)
    message(STATUS "UNITY_PROJECT_ROOT_DIR is not set, skip extansion install target")
endif()

add_subdirectory(pluginInterface)
add_subdirectory(actionInterface)
add_subdirectory(logger)
add_subdirectory(pluginAPI)
add_subdirectory(actions)
add_subdirectory(interopHelper)
