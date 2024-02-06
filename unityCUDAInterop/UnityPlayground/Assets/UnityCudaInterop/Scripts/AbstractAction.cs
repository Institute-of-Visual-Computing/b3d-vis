using System;
using System.Net.NetworkInformation;

namespace B3D
{
	namespace UnityCudaInterop
	{
		public abstract class AbstractRenderingAction
		{

			// Default names for exported functions from native pluin library
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

			private bool isCreated_ = false;

			private bool isInitialized_ = false;

			private int renderEventIdOffset_ = 0;

			#endregion private members

			#region properties
			public bool Isinitialized
			{
				get => isInitialized_;
				protected set => isInitialized_ = value;
			}

			public bool IsCreated
			{
				get => isCreated_;
				protected set => isCreated_ = value;
			}

			public virtual IntPtr RenderEventAndDataFuncPointer { get => renderEventAndDataFuncPointer_; protected set => renderEventAndDataFuncPointer_ = value; }

			protected IntPtr ActionPointer { get => actionPointer_; set => actionPointer_ = value; }

			public int RenderEventIdOffset { get => renderEventIdOffset_; private set => renderEventIdOffset_ = value; }


			#endregion properties

			#region abstract internal dll functions

			protected abstract IntPtr CreateAction();

			public abstract void DestroyAction();

			protected abstract int GetRenderEventIdOffset();

			protected abstract IntPtr GetRenderEventAndDataFunc();

			public abstract void InitializeAction(IntPtr data);

			public abstract void TeardownAction();

			#endregion abstract internal dll functions

			public int MapEventId(int eventId)
			{
				return eventId + RenderEventIdOffset;
			}

			protected AbstractRenderingAction()
			{
				ActionPointer = CreateAction();
				if(ActionPointer != IntPtr.Zero)
				{
					IsCreated = true;
					RenderEventIdOffset = GetRenderEventIdOffset();
					RenderEventAndDataFuncPointer = GetRenderEventAndDataFunc();
				}
			}

			~AbstractRenderingAction()
			{
				IsCreated = false;
				Isinitialized = false;
				
				if (ActionPointer != IntPtr.Zero)
				{
					TeardownAction();
					DestroyAction();
					ActionPointer = IntPtr.Zero;
				}
			}
		}
	}
}
