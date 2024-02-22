using System.Runtime.InteropServices;

using B3D.UnityCudaInterop;
using B3D.UnityCudaInterop.NativeStructs;
using UnityEngine.Rendering;



public class UnityActionNanoRenderer : AbstractUnityRenderAction
{
	#region Native structs for this action

	class ActionNanoRendererRenderEventTypes : RenderEventTypes
	{
		public const int ACTION_RENDER = RenderEventTypes.BASE_ACTION_COUNT + 0;
	}

	#endregion

	private ActionNanoRenderer action_;

	#region AbstractUnityAction Overrides

	protected override AbstractRenderingAction NativeAction
	{
		get { return action_; }
	}

	protected override void InitAction()
	{
		action_ = new();
	}
	protected override void InitRenderingCommandBuffers()
	{
		CommandBuffer cb = new();
		cb.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(ActionNanoRendererRenderEventTypes.ACTION_RENDER), unityRenderingDataPtr);
		renderingCommandBuffers_.Add(new(CameraEvent.BeforeForwardOpaque, cb));
	}

	protected override void FillAdditionalNativeRenderingData()
	{
		// Fill struct with custom data and copy struct to unmanaged code.

		unityRenderingData.volumeTransform.position = volumeCube.transform.position;
		unityRenderingData.volumeTransform.scale = volumeCube.transform.localScale;
		unityRenderingData.volumeTransform.rotation = volumeCube.transform.rotation;
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
