using System.Runtime.InteropServices;

using B3D.UnityCudaInterop;

// This class acts as a template for new actions. Do not change the content in any way! <LINETOREMOVE>

public class UnityRenderActionTemplate : AbstractUnityRenderAction
{
	#region Native structs for this action

	[StructLayout(LayoutKind.Sequential)]
	protected struct TemplateNativeRenderingData
	{
		
	}

	class TemplateRenderEventTypes : RenderEventTypes
	{
		public const int ACTION_RENDER = RenderEventTypes.BASE_ACTION_COUNT + 0;
	}

	#endregion

	private RenderActionTemplate action_;

	protected TemplateNativeRenderingData templateNativeRenderingData_;

	#region AbstractUnityAction Overrides

	protected override AbstractRenderingAction NativeAction
	{
		get { return action_; }
	}

	protected override void InitAction()
	{
		action_ = new();
	}

	protected override void InitAdditionalNativeStruct()
	{
		templateNativeRenderingData_ = new()
		{
			
		};

		additionalNativeStructDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<TemplateNativeRenderingData>());
	}

	protected override void InitRenderingCommandBuffers()
	{
		/*
			CommandBuffer cb = new();
			cb.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(TemplateRenderEventTypes.ACTION_RENDER), renderingActionNativeRenderingDataWrapperPtr_);
			renderingCommandBuffers_.Add(new(CameraEvent.BeforeForwardOpaque, cb));
		*/
	}

	protected override void FillAdditionalNativeRenderingData()
	{
		// Fill struct with custom data and copy struct to unmanaged code.

		Marshal.StructureToPtr(templateNativeRenderingData_, additionalNativeStructDataPtr, true);
	}

	#endregion AbstractUnityAction Overrides


	/// TODO: Current approach is to override and call parent methods like shown below. Not nice. Change to smth other
	#region Unity Methods

	protected override void Start()
	{
		base.Start();
	}

	protected override void Update()
	{
		base.Update();
	}

	protected override void OnDestroy()
	{
		base.OnDestroy();
	}

	#endregion Unity Methods

}
