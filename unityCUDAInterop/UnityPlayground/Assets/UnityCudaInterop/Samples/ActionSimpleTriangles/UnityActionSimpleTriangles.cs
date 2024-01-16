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


public class UnityActionSimpleTriangles : MonoBehaviour
{
    private ActionSimpleTriangles action;
	private ActionTextureProvider actionTextureProvider;

    CommandBuffer commandBuffer;

	public RawImage ri;

	public GameObject volumeCube;

	public Renderer quadRenderer;
	public Material fullscreenMaterial;
	Material quadFullscreenMaterial;

	[StructLayout(LayoutKind.Sequential)]
	struct NativeCameraData
    {
		public Vector3 origin;
		public Vector3 at;
		public Vector3 up;
		public float cosFovY;
		public float fovY;
    }

	[StructLayout(LayoutKind.Sequential)]
	struct NativeUnityTexture
	{
		public IntPtr texturePointer;
		public TextureExtent extent;
	}

	[StructLayout(LayoutKind.Sequential)]
	struct NativeTextureData
	{
		public NativeUnityTexture colorTexture;
		public NativeUnityTexture depthTexture;
		public static NativeTextureData CREATE()
		{
			NativeTextureData ntd = new();
			ntd.colorTexture = new NativeUnityTexture();
			ntd.depthTexture = new NativeUnityTexture();
			return ntd;
		}
	}

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
	struct NativeCube
	{
		public Vector3 position;
		public Vector3 scale;
		public Quaternion rotation;
	};

	[StructLayout(LayoutKind.Sequential)]
    struct NativeRenderingData
    {
        public int eyeCount;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public NativeCameraData[] nativeCameradata;

		public NativeCube nativeCube;
		public static NativeRenderingData CREATE()
		{
			var nrd = new NativeRenderingData();
			nrd.nativeCameradata = new NativeCameraData[2];
			nrd.nativeCube = new();
			return nrd;
		}
	}

	NativeRenderingData nativeRenderingData = new NativeRenderingData();
    System.IntPtr nativeRenderingDataPtr;

	NativeInitData nativeInitData;
	System.IntPtr nativeInitDataPtr;

	NativeTextureData nativeTextureData;
	IntPtr nativeTextureDataPtr;

	Texture2D texture2D;
	Material m;
    // Start is called before the first frame update
    void Start()
    {
		quadFullscreenMaterial = new Material(fullscreenMaterial);
		quadRenderer.material = quadFullscreenMaterial;

		commandBuffer = new ();
		action = new();
		actionTextureProvider = new();
		nativeRenderingData = NativeRenderingData.CREATE();
		nativeInitData = new();
		nativeTextureData = new();

		nativeRenderingDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<NativeRenderingData>());
		nativeInitDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<NativeInitData>());
		nativeTextureDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<NativeTextureData>());

		updateTextures();

		fillNativeRenderingData();
		nativeRenderingData.nativeCube.position = volumeCube.transform.position;
		nativeRenderingData.nativeCube.scale = volumeCube.transform.localScale;
		nativeRenderingData.nativeCube.rotation = volumeCube.transform.rotation;
		Marshal.StructureToPtr(nativeRenderingData, nativeRenderingDataPtr, true);

		commandBuffer.IssuePluginEventAndData(action.RenderEventAndDataFuncPointer, action.mapEventId(2), nativeRenderingDataPtr);
		Camera.main.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, commandBuffer);

		StartCoroutine(InitPluginAtEndOfFrame());
    }

	struct EyeCamera
	{
		public EyeCamera(int eyeIdx, MonoOrStereoscopicEye camEye, XRNode n, InputFeatureUsage<Vector3> feature)
		{
			eyeIndex = eyeIdx;
			cameraEye = camEye;
			xrNode = n;
			nodeUsage = feature;
		}
		public readonly int eyeIndex;
		public readonly Camera.MonoOrStereoscopicEye cameraEye;
		public readonly XRNode xrNode;
		public readonly InputFeatureUsage<Vector3> nodeUsage;
	}

	readonly EyeCamera[] eyeCameraMapping = new EyeCamera[]
	{
		new( 0, Camera.MonoOrStereoscopicEye.Left, XRNode.LeftEye, CommonUsages.leftEyePosition),
		new( 1, Camera.MonoOrStereoscopicEye.Right, XRNode.RightHand, CommonUsages.rightEyePosition)
	};

	void fillNativeRenderingData()
	{
		if (XRSettings.isDeviceActive)
		{
			nativeRenderingData.eyeCount = 2;

			Vector3 cameraPosition = Camera.main.transform.localPosition;
			var centerEyeDevice = InputDevices.GetDeviceAtXRNode(XRNode.CenterEye);

			foreach (var nodeUsage in eyeCameraMapping)
			{
				Camera.MonoOrStereoscopicEye stereoEye = nodeUsage.cameraEye;
				Vector3 eyePos = Vector3.zero;
				if (!centerEyeDevice.TryGetFeatureValue(nodeUsage.nodeUsage, out eyePos))
				{
					stereoEye = MonoOrStereoscopicEye.Left;
					Debug.Log("Could not get position for Node " + nodeUsage.nodeUsage.name);
				}

				Vector3 eyeWorldPos = Camera.main.transform.TransformPoint(eyePos - Camera.main.transform.position);
				setNativeRenderingCameraData(eyeWorldPos, Camera.main.transform.forward, Camera.main.transform.up, Camera.main.fieldOfView, nodeUsage.eyeIndex);
			}
		}
		else
		{
			nativeRenderingData.eyeCount = 1;
			setNativeRenderingCameraData(Camera.main.transform.position, Camera.main.transform.forward, Camera.main.transform.up, Camera.main.fieldOfView, 0);
		}
	}

	void setNativeRenderingCameraData(Vector3 eyePos, Vector3 forward, Vector3 up, float fovYDegree, int eyeIndex)
	{
		nativeRenderingData.nativeCameradata[eyeIndex].origin = eyePos;
		nativeRenderingData.nativeCameradata[eyeIndex].at = eyePos + forward;
		nativeRenderingData.nativeCameradata[eyeIndex].up = up;
		nativeRenderingData.nativeCameradata[eyeIndex].fovY = Mathf.Deg2Rad* fovYDegree;
		nativeRenderingData.nativeCameradata[eyeIndex].cosFovY = Mathf.Cos(nativeRenderingData.nativeCameradata[eyeIndex].fovY);
	}

	void updateTextures()
	{
		nativeTextureData = new();
		actionTextureProvider.createExternalTargetTexture();

		quadFullscreenMaterial.SetTexture("_MainTex", actionTextureProvider.ExternalTargetTexture);

		nativeTextureData.depthTexture.extent.depth = 0;
		nativeTextureData.depthTexture.texturePointer = IntPtr.Zero;

		nativeTextureData.colorTexture.texturePointer = actionTextureProvider.ExternalTargetTexture.GetNativeTexturePtr();
		nativeTextureData.colorTexture.extent = actionTextureProvider.ExternalTargetTextureExtent;

		Marshal.StructureToPtr(nativeTextureData, nativeTextureDataPtr, true);
	}

	private void Update()
	{
		if(actionTextureProvider.renderTextureDescriptorChanged())
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
			nativeRenderingData.nativeCube.position = volumeCube.transform.position;
			nativeRenderingData.nativeCube.scale = volumeCube.transform.localScale;
			nativeRenderingData.nativeCube.rotation = volumeCube.transform.rotation;
			Marshal.StructureToPtr(nativeRenderingData, nativeRenderingDataPtr, true);
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

        Marshal.FreeHGlobal(nativeRenderingDataPtr);
		Marshal.FreeHGlobal(nativeInitDataPtr);
		Marshal.FreeHGlobal(nativeTextureDataPtr);
	}
}
