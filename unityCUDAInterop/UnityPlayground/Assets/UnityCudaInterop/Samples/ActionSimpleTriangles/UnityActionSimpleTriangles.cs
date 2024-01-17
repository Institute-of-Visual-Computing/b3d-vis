using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;

using UnityEngine.Rendering;
using System;
using UnityEngine.UI;
using static System.Net.Mime.MediaTypeNames;
using UnityEngine.XR;
using static UnityEngine.Camera;
using UnityEditor;
using B3D.UnityCudaInterop.NativeStructs;
using JetBrains.Annotations;
using UnityEngine.XR.Management;
using Unity.XR.MockHMD;
using UnityEngine.UIElements;


public class UnityActionSimpleTriangles : MonoBehaviour
{
    private ActionSimpleTriangles action;

    CommandBuffer commandBuffer;

	public RawImage ri;

	public GameObject volumeCube;

	public Renderer quadRenderer;
	public Material fullscreenMaterial;
	Material quadFullscreenMaterial;

	[StructLayout(LayoutKind.Sequential)]
	struct NativeInitData
	{
		public NativeTextureData textureData;
		public static NativeInitData CREATE()
		{
			NativeInitData nid = new();
			nid.textureData = NativeTextureData.CREATE();
			return nid;
		}
	};

	[StructLayout(LayoutKind.Sequential)]
    struct SimpleTriangleNativeRenderingData
	{
		public VolumeTransform nativeCube;
	}

	NativeRenderingDataWrapper nativeRenderingDataWrapper = new NativeRenderingDataWrapper();
	System.IntPtr nativeRenderingDataWrapperPtr;

	SimpleTriangleNativeRenderingData simpleTriangleNativeRenderingData = new SimpleTriangleNativeRenderingData();
	System.IntPtr simpleTriangleNativeRenderingDataPtr;

	NativeInitData nativeInitData;
	System.IntPtr nativeInitDataPtr;

	NativeTextureData nativeTextureData;
	IntPtr nativeTextureDataPtr;

    // Start is called before the first frame update
    void Start()
    {
		quadFullscreenMaterial = new Material(fullscreenMaterial);
		quadRenderer.material = quadFullscreenMaterial;

		commandBuffer = new ();
		action = new();
		nativeRenderingDataWrapper = NativeRenderingDataWrapper.CREATE();
		simpleTriangleNativeRenderingData = new SimpleTriangleNativeRenderingData();
		
		nativeInitData = new();
		nativeTextureData = new();

		nativeInitDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<NativeInitData>());
		nativeTextureDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<NativeTextureData>());
		nativeRenderingDataWrapperPtr = Marshal.AllocHGlobal(Marshal.SizeOf<NativeRenderingDataWrapper>());
		simpleTriangleNativeRenderingDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<SimpleTriangleNativeRenderingData>());
		nativeRenderingDataWrapper.AdditionalDataPointer = simpleTriangleNativeRenderingDataPtr;

		updateTextures();

		fillNativeRenderingData();
		simpleTriangleNativeRenderingData.nativeCube.position = volumeCube.transform.position;
		simpleTriangleNativeRenderingData.nativeCube.scale = volumeCube.transform.localScale;
		simpleTriangleNativeRenderingData.nativeCube.rotation = volumeCube.transform.rotation;
		Marshal.StructureToPtr(simpleTriangleNativeRenderingData, simpleTriangleNativeRenderingDataPtr, true);
		Marshal.StructureToPtr(nativeRenderingDataWrapper, nativeRenderingDataWrapperPtr, true);

		commandBuffer.IssuePluginEventAndData(action.RenderEventAndDataFuncPointer, action.mapEventId(2), nativeRenderingDataWrapperPtr);
		Camera.main.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, commandBuffer);

		StartCoroutine(InitPluginAtEndOfFrame());
    }

	struct EyeCamera
	{
		public EyeCamera(int eyeIdx, StereoscopicEye camEye, XRNode n, InputFeatureUsage<Vector3> feature)
		{
			eyeIndex = eyeIdx;
			cameraEye = camEye;
			xrNode = n;
			nodeUsage = feature;
		}
		public readonly int eyeIndex;
		public readonly Camera.StereoscopicEye cameraEye;
		public readonly XRNode xrNode;
		public readonly InputFeatureUsage<Vector3> nodeUsage;
	}

	readonly EyeCamera[] eyeCameraMapping = new EyeCamera[]
	{
		new( 0, Camera.StereoscopicEye.Left, XRNode.LeftEye, CommonUsages.leftEyePosition),
		new( 1, Camera.StereoscopicEye.Right, XRNode.RightHand, CommonUsages.rightEyePosition)
	};

	void fillNativeRenderingData()
	{
		if (XRSettings.isDeviceActive)
		{
			nativeRenderingDataWrapper.NativeRenderingData.EyeCount = 2;

			var centerEyeDevice = InputDevices.GetDeviceAtXRNode(XRNode.CenterEye);
			Vector3 cameraWorldPosition;
			if(!centerEyeDevice.TryGetFeatureValue(CommonUsages.centerEyePosition, out cameraWorldPosition))
			{
				 cameraWorldPosition = Camera.main.transform.localPosition;
			}
			cameraWorldPosition = Camera.main.transform.position;

			var displaySubSytems = new List<XRDisplaySubsystem>();
			SubsystemManager.GetInstances<XRDisplaySubsystem>(displaySubSytems);
			XRDisplaySubsystem.XRRenderPass renderpass;
			bool renderpassAvailable = false;
			if (displaySubSytems.Count > 0)
			{
				renderpassAvailable = true;
			}


			XRDisplaySubsystem.XRRenderParameter[] renderParameter = new XRDisplaySubsystem.XRRenderParameter[2];


			foreach (var nodeUsage in eyeCameraMapping)
			{
				Camera.StereoscopicEye stereoEye = nodeUsage.cameraEye;
				Vector3 eyePos = Vector3.zero;
				if (!centerEyeDevice.TryGetFeatureValue(nodeUsage.nodeUsage, out eyePos))
				{
					var viewMat = Camera.main.GetStereoViewMatrix(stereoEye);
					var camMat = viewMat.inverse;
					eyePos = cameraWorldPosition + Camera.main.transform.right * Camera.main.stereoSeparation * 0.5f * (stereoEye == StereoscopicEye.Left ? -1.0f : 1.0f);
				}

				Vector3 eyeWorldPos = Camera.main.transform.TransformPoint(eyePos);
				
				setNativeRenderingCameraData(eyePos, Camera.main.transform.forward, Camera.main.transform.up, Camera.main.fieldOfView, nodeUsage.eyeIndex);
				if (renderpassAvailable && displaySubSytems[0].GetRenderPassCount() > 0)
				{

					var upperLeft = Camera.main.ScreenToWorldPoint(new Vector3(0, action.TextureProvider.ExternalTargetTextureExtent.Height - 1, 1), (MonoOrStereoscopicEye)nodeUsage.cameraEye);
					var upperRight = Camera.main.ScreenToWorldPoint(new Vector3(action.TextureProvider.ExternalTargetTextureExtent.Width - 1, action.TextureProvider.ExternalTargetTextureExtent.Height - 1, 1), (MonoOrStereoscopicEye)nodeUsage.cameraEye);
					var lowerLeft = Camera.main.ScreenToWorldPoint(new Vector3(0, 0, 1), (MonoOrStereoscopicEye)nodeUsage.cameraEye);


					var onePxDirectionU = (upperRight - upperLeft);// / action.TextureProvider.ExternalTargetTextureExtent.Width;
					var onePxDirectionV = (upperLeft - lowerLeft);//  / action.TextureProvider.ExternalTargetTextureExtent.Height;
					var camLowerLeft = (lowerLeft - eyePos);

					nativeRenderingDataWrapper.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].dir00 = camLowerLeft;
					nativeRenderingDataWrapper.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].dirDu = onePxDirectionU;
					nativeRenderingDataWrapper.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].dirDv = onePxDirectionV;
					nativeRenderingDataWrapper.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].directionsAvailable = true;

				}
			}
		}
		else
		{
			nativeRenderingDataWrapper.NativeRenderingData.EyeCount = 1;
			setNativeRenderingCameraData(Camera.main.transform.position, Camera.main.transform.forward, Camera.main.transform.up, Camera.main.fieldOfView, 0);

		}
	}

	void setNativeRenderingCameraData(Vector3 eyePos, Vector3 forward, Vector3 up, float fovYDegree, int eyeIndex)
	{
		NativeCameraData nativeCameraData = new()
		{
			Origin = eyePos,
			At = Camera.main.transform.position + forward * Camera.main.stereoConvergence,
			Up = up,
			CosFovY = Mathf.Cos(Mathf.Deg2Rad * fovYDegree),
			FovY = Mathf.Deg2Rad * fovYDegree,
			directionsAvailable = false
		};
		nativeRenderingDataWrapper.NativeRenderingData.NativeCameradata[eyeIndex] = nativeCameraData;
	}

	void updateTextures()
	{
		nativeTextureData = new();
		action.TextureProvider.createExternalTargetTexture();

		quadFullscreenMaterial.SetTexture("_MainTex", action.TextureProvider.ExternalTargetTexture);

		nativeTextureData.DepthTexture.Extent.Depth = 0;
		nativeTextureData.DepthTexture.TexturePointer = IntPtr.Zero;

		nativeTextureData.ColorTexture.TexturePointer = action.TextureProvider.ExternalTargetTexture.GetNativeTexturePtr();
		nativeTextureData.ColorTexture.Extent = action.TextureProvider.ExternalTargetTextureExtent;

		Marshal.StructureToPtr(nativeTextureData, nativeTextureDataPtr, true);
	}

	private void Update()
	{
		if(action.TextureProvider.renderTextureDescriptorChanged())
		{
			updateTextures();
			Camera.main.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, commandBuffer);

			CommandBuffer cbImmediate = new();
			cbImmediate.IssuePluginEventAndData(action.RenderEventAndDataFuncPointer, action.mapEventId(1), nativeTextureDataPtr);
			Graphics.ExecuteCommandBuffer(cbImmediate);
			StartCoroutine(UpdateTextureAfterWait());
		}
		else
		{
			fillNativeRenderingData();
			simpleTriangleNativeRenderingData.nativeCube.position = volumeCube.transform.position;
			simpleTriangleNativeRenderingData.nativeCube.scale = volumeCube.transform.localScale;
			simpleTriangleNativeRenderingData.nativeCube.rotation = volumeCube.transform.rotation;
			Marshal.StructureToPtr(simpleTriangleNativeRenderingData, simpleTriangleNativeRenderingDataPtr, true);
			Marshal.StructureToPtr(nativeRenderingDataWrapper, nativeRenderingDataWrapperPtr, true);
		}
	}

	IEnumerator UpdateTextureAfterWait()
	{
		yield return new WaitForEndOfFrame();
		yield return new WaitForSeconds(1);
		Camera.main.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, commandBuffer);
	}

	IEnumerator InitPluginAtEndOfFrame()
	{
		yield return new WaitForEndOfFrame();

		nativeInitData = new();
		nativeInitData.textureData = nativeTextureData;
		Marshal.StructureToPtr(nativeInitData, nativeInitDataPtr, true);

		CommandBuffer immediate = new();
		immediate.IssuePluginEventAndData(action.RenderEventAndDataFuncPointer, action.mapEventId(0), nativeInitDataPtr);
		Graphics.ExecuteCommandBuffer(immediate);
	}

	void OnDestroy()
	{
		action.teardownAction();
		action.destroyAction();

		Marshal.FreeHGlobal(simpleTriangleNativeRenderingDataPtr);
        Marshal.FreeHGlobal(nativeRenderingDataWrapperPtr);
		Marshal.FreeHGlobal(nativeInitDataPtr);
		Marshal.FreeHGlobal(nativeTextureDataPtr);
	}
}
