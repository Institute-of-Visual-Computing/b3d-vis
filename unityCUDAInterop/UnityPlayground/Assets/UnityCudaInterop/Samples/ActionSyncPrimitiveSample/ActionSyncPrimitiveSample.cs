using System;
using System.Runtime.InteropServices;
using static ActionTest;
using UnityEngine.Rendering;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

public class ActionSyncPrimitiveSample : Action
{
	public class ActionSyncPrimitiveSampleRenderEventIDS : ActionRenderEventIDS
	{
		public const int ACTION_USER_UPDATE = ActionRenderEventIDS.NATIVE_ACTION_RENDER_EVENT_ID_COUNT;

		public const int ACTION_SYNC_PRIMITIVE_SAMPLE_RENDERER_EVENT_ID_COUNT = ACTION_USER_UPDATE + 1;
	}

	const string dllName = "ActionSyncPrimitiveSample";

	[DllImport(dllName, EntryPoint = "createAction")]
	private static extern IntPtr createActionExtern();


	[DllImport(dllName, EntryPoint = "destroyAction")]
	private static extern void destroyActionExtern(IntPtr nativeAction);

	[DllImport(dllName, EntryPoint = "getRenderEventIDOffset")]
	private static extern int getRenderEventIDOffsetExtern(IntPtr nativeAction);

	private Texture2D tex;

	CommandBuffer cb;

	protected override IntPtr create()
	{
		return createActionExtern();
	}

	protected override void destroy(IntPtr actionPointer)
	{
		destroyActionExtern(actionPointer);
	}

	protected override void initialize()
	{
		actionRuntime.ExecuteImmediate(getRenderEventID(ActionRenderEventIDS.ACTION_INITIALIZE), tex.GetNativeTexturePtr());
	}

	protected override int getRenderEventIdOffset()
	{
		return getRenderEventIDOffsetExtern(ActionPointer);
	}
	
	protected override void ActionRegisterDone()
	{
		var lala = getRenderEventID(NativeActionTestRenderEventIDS.ACTION_TEST_UPDATE);
		Debug.Log("RenderEventIdOffset: " + lala);
		cb.IssuePluginEventAndData(actionRuntime.RenderEventAndDataFuncPointer, getRenderEventID(ActionSyncPrimitiveSampleRenderEventIDS.ACTION_USER_UPDATE), IntPtr.Zero);
		Camera.main.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, cb);
	}

	protected override void Start()
	{
		base.Start();

		tex = new Texture2D(1024, 1024, TextureFormat.ARGB32, false);
		tex.Apply();
		cb = new CommandBuffer();

		Debug.Log("SetupAction");
		SetupAction();
	}

	void Update()
	{
	}
}
