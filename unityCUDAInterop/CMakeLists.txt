set(UNITY_PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/UnityPlayground")
set(UNITY_ASSETS_SUBDIRECTORY_RUNTIME "UnityCudaInterop/Runtime")
set(UNITY_ASSET_SUBDIRECTORY_INTEROP_ACTION_SAMPLES "UnityCudaInterop/Samples")
set(UNITY_ASSET_SUBDIRECTORY_INTEROP_ACTIONS "UnityCudaInterop/Actions")


add_subdirectory(pluginInterface)
add_subdirectory(actionInterface)

add_subdirectory(logger)
# add_subdirectory(runtime)
add_subdirectory(pluginAPI)

add_subdirectory(actions)


add_subdirectory(interopHelper)


