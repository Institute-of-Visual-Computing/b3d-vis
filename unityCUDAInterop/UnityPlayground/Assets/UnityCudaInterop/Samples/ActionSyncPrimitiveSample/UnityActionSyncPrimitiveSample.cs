using System.Runtime.InteropServices;

using UnityEngine.Rendering;
using B3D.UnityCudaInterop;
using B3D.UnityCudaInterop.NativeStructs;

public class UnityActionSyncPrimitiveSample : AbstractUnityRenderAction
{

	#region Native structs for this action

	class SyncPrimitiveSampleRenderEventTypes : RenderEventTypes
	{
		public const int ACTION_RENDER = RenderEventTypes.BASE_ACTION_COUNT + 0;
	}

	#endregion

	private ActionSyncPrimitiveSample action;

	#region AbstractUnityAction Overrides

	protected override AbstractRenderingAction NativeAction
	{
		get { return action; }
	}

	protected override void InitAction()
	{
		action = new();
	}

	protected override void InitRenderingCommandBuffers()
	{
		CommandBuffer cb = new();
		cb.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(SyncPrimitiveSampleRenderEventTypes.ACTION_RENDER), unityRenderingDataPtr);
		renderingCommandBuffers_.Add(new(CameraEvent.BeforeForwardOpaque, cb));
	}

	protected override void FillAdditionalNativeRenderingData()
	{
		// Marshal.StructureToPtr(syncPrimitiveSampleNativeRenderingData, additionalNativeStructDataPtr, true);
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
