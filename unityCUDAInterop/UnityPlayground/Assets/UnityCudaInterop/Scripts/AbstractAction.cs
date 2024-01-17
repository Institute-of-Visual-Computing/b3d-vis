using System;
using UnityEngine;

using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using System.Runtime.InteropServices;

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

	private ActionTextureProvider textureProvider_;

	#endregion private members

	#region properties
	public bool Isinitialized
	{
		get => isInitialized_;
		protected set => isInitialized_ = value;
	}

	public virtual IntPtr RenderEventAndDataFuncPointer { get => renderEventAndDataFuncPointer_; protected set => renderEventAndDataFuncPointer_ = value; }

	protected IntPtr ActionPointer { get => actionPointer_; set => actionPointer_ = value; }

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

	protected AbstractAction()
	{
		textureProvider_ = new();
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
	}
}
