using System;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;
using UnityEngine;

public class ActionTest : Action
{
    public class NativeActionTestRenderEventIDS : ActionRenderEventIDS
    {
        public const int ACTION_TEST_UPDATE = NATIVE_ACTION_RENDER_EVENT_ID_COUNT;

        public const int NATIVE_ACTION_TEST_RENDER_EVENT_ID_COUNT = ACTION_TEST_UPDATE + 1;
    }

    const string dllName = "ActionTest";

    [DllImport(dllName, EntryPoint = "createAction")]
    private static extern IntPtr createActionExtern();


    [DllImport(dllName, EntryPoint = "destroyAction")]
    private static extern void destroyActionExtern(IntPtr nativeAction);

    [DllImport(dllName, EntryPoint = "getRenderEventIDOffset")]
    private static extern int getRenderEventIDOffsetExtern(IntPtr nativeAction);

    private Texture2D textureTest;


    protected override IntPtr create()
    {
        return createActionExtern();
    }

    protected override void initialize()
    {
        actionRuntime.ExecuteImmediate(getRenderEventID(ActionRenderEventIDS.ACTION_INITIALIZE), textureTest.GetNativeTexturePtr());
    }

    protected override void destroy(IntPtr actionPointer)
    {
        destroyActionExtern(actionPointer);
    }

    protected override int getRenderEventIdOffset()
    {
        return getRenderEventIDOffsetExtern(ActionPointer);
    }

    struct DataToPlugin
    {
        int value;
        Vector3 bla;
    }

    CommandBuffer cb;
    IntPtr ptrToData;

    protected override void ActionRegisterDone()
    {
		cb.IssuePluginEventAndData(actionRuntime.RenderEventAndDataFuncPointer, getRenderEventID(NativeActionTestRenderEventIDS.ACTION_TEST_UPDATE), IntPtr.Zero);
		Camera.main.AddCommandBuffer(CameraEvent.AfterDepthTexture, cb);
	}

    protected override void Start()
    {
        base.Start();
        cb = new();

        textureTest = new Texture2D(200, 200, TextureFormat.ARGB32, false);
        textureTest.Apply();

        SetupAction();
        

    }
}
