add_library(UnityCUDAInteropPluginAPI INTERFACE)

target_include_directories(UnityCUDAInteropPluginAPI
    INTERFACE
        "include"
)   

add_library(B3D::UnityCUDAInterop::PluginAPI ALIAS UnityCUDAInteropPluginAPI)
