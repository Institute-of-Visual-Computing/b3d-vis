#include <optix_device.h>
#include <owl/owl.h>

#include "SharedStructs.h"
#include "owl/owl_device.h"

using namespace  b3d::renderer::nano;

extern "C" __constant__ LaunchParams optixLaunchParams;

OPTIX_RAYGEN_PROGRAM(rayGeneration)()
{
	
}



