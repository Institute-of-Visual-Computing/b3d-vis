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
using B3D.UnityCudaInterop;

public class AbstractUnityAction : MonoBehaviour
{

}




public class UnityActionSimpleTriangles : MonoBehaviour
{
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

	private ActionSimpleTriangles action;

	private CommandBuffer commandBuffer;

	public Camera targetCamera;

	public RawImage ri;

	public GameObject volumeCube;

	public Renderer quadRenderer;
	public Material fullscreenMaterial;
	Material quadFullscreenMaterial;

	public Material objectMaterial;
	Material volumeObjectMaterial;
	public Renderer objectRenderer;

	SimpleTriangleNativeRenderingData simpleTriangleNativeRenderingData;
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


		volumeObjectMaterial = new Material(objectMaterial);
		objectRenderer.material = volumeObjectMaterial;

		commandBuffer = new ();
		action = new();
		simpleTriangleNativeRenderingData = new();
		
		nativeInitData = new();
		nativeTextureData = new();

		nativeInitDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<NativeInitData>());
		nativeTextureDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<NativeTextureData>());
		simpleTriangleNativeRenderingDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<SimpleTriangleNativeRenderingData>());
		action.setAdditionalDataPointer(simpleTriangleNativeRenderingDataPtr);
		action.setCamera(targetCamera);

		updateTextures();
		fillNativeRenderingData();

		commandBuffer.IssuePluginEventAndData(action.RenderEventAndDataFuncPointer, action.mapEventId(2), action.NativeRenderingDataWrapperPointer);

		targetCamera.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, commandBuffer);

		StartCoroutine(InitPluginAtEndOfFrame());
    }

	void fillNativeRenderingData()
	{
		action.fillNativeRenderingDataWrapper();
		simpleTriangleNativeRenderingData.nativeCube.position = volumeCube.transform.position;
		simpleTriangleNativeRenderingData.nativeCube.scale = volumeCube.transform.localScale;
		simpleTriangleNativeRenderingData.nativeCube.rotation = volumeCube.transform.rotation;
		Marshal.StructureToPtr(simpleTriangleNativeRenderingData, simpleTriangleNativeRenderingDataPtr, true);
	}
	
	void updateTextures()
	{
		nativeTextureData = new();
		action.TextureProvider.createExternalTargetTexture();

		quadFullscreenMaterial.SetTexture("_MainTex", action.TextureProvider.ExternalTargetTexture);
		volumeObjectMaterial.SetTexture("_MainTex", action.TextureProvider.ExternalTargetTexture);

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
			targetCamera.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, commandBuffer);

			CommandBuffer cbImmediate = new();
			cbImmediate.IssuePluginEventAndData(action.RenderEventAndDataFuncPointer, action.mapEventId(1), nativeTextureDataPtr);
			Graphics.ExecuteCommandBuffer(cbImmediate);
			StartCoroutine(UpdateTextureAfterWait());
		}
		else
		{
			fillNativeRenderingData();
		}
	}

	IEnumerator UpdateTextureAfterWait()
	{
		yield return new WaitForEndOfFrame();
		yield return new WaitForSeconds(1);
		targetCamera.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, commandBuffer);
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
		Marshal.FreeHGlobal(nativeInitDataPtr);
		Marshal.FreeHGlobal(nativeTextureDataPtr);
	}
}
