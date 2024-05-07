using B3D.UnityCudaInterop;
using B3D.UnityCudaInterop.NativeStructs;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

[StructLayout(LayoutKind.Sequential)]
public struct ActionNativeTextureData
{
	public UnityExtent extent;
	public IntPtr nativeTexturePointer;
}

public class UnityActionNativeTexture : MonoBehaviour
{
	public bool copyBack = false;

	public Texture2D externalTex;
	class ActionNativeTextureEventTypes : RenderEventTypes
	{
		public const int ACTION_RENDER = RenderEventTypes.BASE_ACTION_COUNT + 0;
	}

	private ActionNativeTexture action_;

	protected ActionNativeTextureData actionData;

	public RawImage image;


	protected System.IntPtr actionDataPtr;

	CommandBuffer renderBuffer;

	protected virtual void Start()
	{
		InitAllObjects();
		actionData.extent = new UnityExtent(256, 256, 1);
		StartCoroutine(InitPluginAtEndOfFrame());
	}

	protected AbstractRenderingAction NativeAction => action_;
	protected void InitAction()
	{
		action_ = new ActionNativeTexture();
	}

	protected virtual IEnumerator InitPluginAtEndOfFrame()
	{
		yield return new WaitForEndOfFrame();

		Marshal.StructureToPtr(actionData, actionDataPtr, true);

		renderBuffer = new();
		renderBuffer.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(RenderEventTypes.BASE_ACTION_COUNT), actionDataPtr);

		CommandBuffer immediate = new();
		immediate.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(RenderEventTypes.ACTION_INITIALIZE), actionDataPtr);

		addRendercmd();

		yield return new WaitForEndOfFrame();
		yield return new WaitForEndOfFrame();
	}

	protected virtual void InitAllObjects()
	{
		actionData = new();
		actionData.extent = new();
		actionData.nativeTexturePointer = IntPtr.Zero;

		actionDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<ActionNativeTextureData>());
		Debug.Log(Marshal.SizeOf<ActionNativeTextureData>());
		InitAction();
	}



	private void OnDestroy()
	{
		image.texture = null;
		
		Destroy(image);
		Camera.main.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, renderBuffer);
		Destroy(externalTex);
		action_.TeardownAction();
		action_.DestroyAction();
	}

	public void createTex()
	{
		actionData = Marshal.PtrToStructure<ActionNativeTextureData>(actionDataPtr);
		image.texture = null;
		externalTex = null;
		externalTex = Texture2D.CreateExternalTexture(256, 256, TextureFormat.RGBA32, false, true, actionData.nativeTexturePointer);
	}

	public void addRendercmd()
	{
		Camera.main.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, renderBuffer);
	}
	public void removeRenderCmd()
	{
		Camera.main.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, renderBuffer);
	}

	public void setTex2Img()
	{
		image.texture = externalTex;
	}
}
