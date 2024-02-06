using System;
using System.Runtime.InteropServices;
namespace B3D
{
	namespace UnityCudaInterop
	{
		public class ActionSimpleTriangles : AbstractRenderingAction
		{
			#region dll function signatures

			const string dllName = "ActionSimpleTriangles";

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
			protected override IntPtr CreateAction()
			{
				return createActionExtern();
			}

			public override void DestroyAction()
			{
				destroyActionExtern(ActionPointer);
				ActionPointer = IntPtr.Zero;
			}

			public override void InitializeAction(IntPtr data)
			{
				initializeActionExtern(ActionPointer, data);
			}

			public override void TeardownAction()
			{
				teardownActionExtern(ActionPointer);
			}

			protected override int GetRenderEventIdOffset()
			{
				return getRenderEventIDOffsetExtern(ActionPointer);
			}

			protected override IntPtr GetRenderEventAndDataFunc()
			{
				return getRenderEventAndDataFuncExtern();
			}

			#endregion dll function calls

			public ActionSimpleTriangles() : base()
			{

			}
		}
	}
}
