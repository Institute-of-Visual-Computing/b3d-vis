#pragma once

#define EXTERN_CREATE_ACTION(classname)																				\
	extern "C" UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API CreateAction()->b3d::unity_cuda_interop::Action*		\
	{																												\
		b3d::unity_cuda_interop::Action* a = new classname;													        \
		sPluginHandler.registerAction(a);																			\
		return a;																									\
	}

