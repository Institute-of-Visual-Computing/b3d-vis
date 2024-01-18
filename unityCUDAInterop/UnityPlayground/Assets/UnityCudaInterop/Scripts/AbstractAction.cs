using System;
using UnityEngine;

using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using System.Runtime.InteropServices;
using B3D.UnityCudaInterop.NativeStructs;
using B3D.UnityCudaInterop;
using static UnityEngine.Camera;
using UnityEngine.XR;

public abstract class AbstractAction
{
	public const int eventIdCount = 10;

	#region DLL exported function names

	protected const string dllFuncNameCreateAction = "CreateAction";
	protected const string dllFuncNameDestroyAction = "DestroyAction";
	protected const string dllFuncNameGetRenderEventIDOffset = "GetRenderEventIDOffset";
	protected const string dllFuncNameInitializeAction = "InitializeAction";
	protected const string dllFuncNameTeardownAction = "TeardownAction";
	protected const string dllFuncNameGetRenderEventAndDataFunc = "GetRenderEventAndDataFunc";

	#endregion DLL exported function names

	#region private members

	private IntPtr actionPointer_ = IntPtr.Zero;

	protected IntPtr renderEventAndDataFuncPointer_ = IntPtr.Zero;

	private bool isInitialized_ = false;
	private int renderEventIdOffset_ = 0;

	private Camera targetCamera_;

	private ActionTextureProvider textureProvider_;

	NativeRenderingDataWrapper nativeRenderingDataWrapper_;
	System.IntPtr nativeRenderingDataWrapperPtr_;

	#endregion private members

	#region properties
	public bool Isinitialized
	{
		get => isInitialized_;
		protected set => isInitialized_ = value;
	}

	public virtual IntPtr RenderEventAndDataFuncPointer { get => renderEventAndDataFuncPointer_; protected set => renderEventAndDataFuncPointer_ = value; }

	protected IntPtr ActionPointer { get => actionPointer_; set => actionPointer_ = value; }

	public IntPtr NativeRenderingDataWrapperPointer {  get => nativeRenderingDataWrapperPtr_; }

	protected int RenderEventIdOffset { get => renderEventIdOffset_; set => renderEventIdOffset_ = value; }

	public ActionTextureProvider TextureProvider { get => textureProvider_; }


	#endregion properties

	#region abstract internal dll functions

	protected abstract IntPtr createAction();

	public abstract void destroyAction();

	protected abstract int getRenderEventIdOffset();

	protected abstract IntPtr getRenderEventAndDataFunc();

	public abstract void initializeAction(IntPtr data);

	public abstract void teardownAction();

	#endregion abstract internal dll functions


	public int mapEventId(int eventId)
	{
		return eventId + RenderEventIdOffset;
	}

	public void setAdditionalDataPointer(IntPtr data)
	{
		nativeRenderingDataWrapper_.AdditionalDataPointer = data;
	}

	public void setCamera(Camera camera)
	{
		this.targetCamera_ = camera;
	}

	public void fillNativeRenderingDataWrapper()
	{
		if (XRSettings.isDeviceActive)
		{
			nativeRenderingDataWrapper_.NativeRenderingData.EyeCount = 2;

			Vector3 cameraWorldPosition = targetCamera_.transform.position;

			XRDisplaySubsystem.XRRenderParameter[] renderParameter = new XRDisplaySubsystem.XRRenderParameter[2];

			foreach (var nodeUsage in SharedMembers.eyeCameraMapping)
			{
				Camera.StereoscopicEye stereoEye = nodeUsage.cameraEye;
				
				var eyePos = cameraWorldPosition + targetCamera_.transform.right * targetCamera_.stereoSeparation * 0.5f * (stereoEye == StereoscopicEye.Left ? -1.0f : 1.0f);
				
				setNativeRenderingCameraData(eyePos, targetCamera_.transform.forward + eyePos, targetCamera_.transform.up, targetCamera_.fieldOfView, nodeUsage.eyeIndex);

				var upperLeft = targetCamera_.ScreenToWorldPoint(new Vector3(0, TextureProvider.ExternalTargetTextureExtent.Height - 1, 1), (MonoOrStereoscopicEye)nodeUsage.cameraEye);
				var upperRight = targetCamera_.ScreenToWorldPoint(new Vector3(TextureProvider.ExternalTargetTextureExtent.Width - 1, TextureProvider.ExternalTargetTextureExtent.Height - 1, 1), (MonoOrStereoscopicEye)nodeUsage.cameraEye);
				var lowerLeft = targetCamera_.ScreenToWorldPoint(new Vector3(0, 0, 1), (MonoOrStereoscopicEye)nodeUsage.cameraEye);

				var onePxDirectionU = (upperRight - upperLeft);// / action.TextureProvider.ExternalTargetTextureExtent.Width;
				var onePxDirectionV = (upperLeft - lowerLeft);//  / action.TextureProvider.ExternalTargetTextureExtent.Height;
				var camLowerLeft = (lowerLeft - eyePos);

				nativeRenderingDataWrapper_.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].dir00 = camLowerLeft;
				nativeRenderingDataWrapper_.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].dirDu = onePxDirectionU;
				nativeRenderingDataWrapper_.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].dirDv = onePxDirectionV;
				nativeRenderingDataWrapper_.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].directionsAvailable = true;
			}
		}
		else
		{
			nativeRenderingDataWrapper_.NativeRenderingData.EyeCount = 1;
			setNativeRenderingCameraData(targetCamera_.transform.position, targetCamera_.transform.forward, targetCamera_.transform.up, targetCamera_.fieldOfView, 0);
		}
		Marshal.StructureToPtr(nativeRenderingDataWrapper_, nativeRenderingDataWrapperPtr_, true);
	}

	void setNativeRenderingCameraData(Vector3 origin, Vector3 at, Vector3 up, float fovYDegree, int eyeIndex)
	{
		NativeCameraData nativeCameraData = new()
		{
			Origin = origin,
			At = at,
			Up = up,
			CosFovY = Mathf.Cos(Mathf.Deg2Rad * fovYDegree),
			FovY = Mathf.Deg2Rad * fovYDegree,
			directionsAvailable = false
		};
		nativeRenderingDataWrapper_.NativeRenderingData.NativeCameradata[eyeIndex] = nativeCameraData;
	}




	protected AbstractAction()
	{
		textureProvider_ = new();
		nativeRenderingDataWrapper_ = NativeRenderingDataWrapper.CREATE();
		nativeRenderingDataWrapperPtr_ = Marshal.AllocHGlobal(Marshal.SizeOf<NativeRenderingDataWrapper>());
		ActionPointer = createAction();
		RenderEventIdOffset = getRenderEventIdOffset();
		RenderEventAndDataFuncPointer = getRenderEventAndDataFunc();
	}

	~AbstractAction()
	{
		if(ActionPointer != IntPtr.Zero)
		{
			teardownAction();
			destroyAction();
		}
		Marshal.FreeHGlobal(nativeRenderingDataWrapperPtr_);
	}
}
