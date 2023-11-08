add_library(UnityAPI INTERFACE)

target_include_directories(UnityAPI
    INTERFACE
        "include"
)   

add_library(UnityCUDAInterop::UnityAPI ALIAS UnityAPI)
