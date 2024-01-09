using System;
using System.Runtime.InteropServices;

public class ActionConcrete : AbstractAction
{
	#region dll function signatures
	const string dllName = "ActionConcrete";
	
	[DllImport(dllName, EntryPoint = dllFuncNameCreateAction)]
	private static extern IntPtr createActionExtern();


	[DllImport(dllName, EntryPoint = dllFuncNameDestroyAction)]
	private static extern void destroyActionExtern(IntPtr nativeAction);


	[DllImport(dllName, EntryPoint = dllFuncNameGetRenderEventIDOffset)]
	private static extern int getRenderEventIDOffsetExtern(IntPtr nativeAction);


	[DllImport(dllName, EntryPoint = dllFuncNameInitializeAction)]
	private static extern void initializeActionExtern(IntPtr nativeAction, IntPtr data);


	[DllImport(dllName, EntryPoint = dllFuncNameTeardownAction)]
	private static extern void teardownActionExtern(IntPtr nativeAction);

	[DllImport(dllName, EntryPoint = dllFuncNameGetRenderEventAndDataFunc)]
	private static extern IntPtr getRenderEventAndDataFuncExtern();

	#endregion dll function signatures

	#region dll function calls

	protected override IntPtr createAction()
	{
		return createActionExtern();
	}

	public override void destroyAction()
	{
		destroyActionExtern(ActionPointer);
		ActionPointer = IntPtr.Zero;
	}

	protected override int getRenderEventIdOffset()
	{
		return getRenderEventIDOffsetExtern(ActionPointer);
	}


	public override void initializeAction(IntPtr data)
	{
		initializeActionExtern(ActionPointer, data);
	}

	public override void teardownAction()
	{
		teardownActionExtern(ActionPointer);
	}

	protected override IntPtr getRenderEventAndDataFunc()
	{
		return getRenderEventAndDataFuncExtern();
	}

	#endregion dll function calls


	public ActionConcrete() : base()
	{

	}

	public void initialize(IntPtr data)
	{
		initialize(data);
	}

}
