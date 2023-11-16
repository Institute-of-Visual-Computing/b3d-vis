add_library(UnityCUDAInteropUnityPluginAPI INTERFACE)

target_include_directories(UnityCUDAInteropUnityPluginAPI
    INTERFACE
        "include"
)   

add_library(UnityCUDAInterop::UnityPluginAPI ALIAS UnityCUDAInteropUnityPluginAPI)
