using System.Runtime.InteropServices;

using B3D.UnityCudaInterop;
using B3D.UnityCudaInterop.NativeStructs;
using UnityEngine.Rendering;



public class UnityActionNanoRenderer : AbstractUnityRenderAction
{
	#region Native structs for this action

	[StructLayout(LayoutKind.Sequential)]
	protected struct ActionNanoRendererNativeRenderingData
	{
		public VolumeTransform volumeTransform;
	}

	class ActionNanoRendererRenderEventTypes : RenderEventTypes
	{
		public const int ACTION_RENDER = RenderEventTypes.BASE_ACTION_COUNT + 0;
	}

	#endregion

	private ActionNanoRenderer action_;

	protected ActionNanoRendererNativeRenderingData actionNanoRendererNativeRenderingData_;

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
		actionNanoRendererNativeRenderingData_ = new()
		{
			volumeTransform = new()
		};

		additionalNativeStructDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<ActionNanoRendererNativeRenderingData>());
	}

	protected override void InitRenderingCommandBuffers()
	{
		CommandBuffer cb = new();
		cb.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(ActionNanoRendererRenderEventTypes.ACTION_RENDER), renderingActionNativeRenderingDataWrapperPtr_);
		renderingCommandBuffers_.Add(new(CameraEvent.BeforeForwardOpaque, cb));
	}

	protected override void FillAdditionalNativeRenderingData()
	{
		// Fill struct with custom data and copy struct to unmanaged code.

		actionNanoRendererNativeRenderingData_.volumeTransform.position = volumeCube.transform.position;
		actionNanoRendererNativeRenderingData_.volumeTransform.scale = volumeCube.transform.localScale;
		actionNanoRendererNativeRenderingData_.volumeTransform.rotation = volumeCube.transform.rotation;

		Marshal.StructureToPtr(actionNanoRendererNativeRenderingData_, additionalNativeStructDataPtr, true);
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
