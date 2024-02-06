using System.Runtime.InteropServices;

using UnityEngine.Rendering;
using B3D.UnityCudaInterop.NativeStructs;
using B3D.UnityCudaInterop;

public class UnityActionSimpleTriangles : AbstractUnityRenderAction
{
	#region Native structs for this action

	[StructLayout(LayoutKind.Sequential)]
	protected struct SimpleTrianglesNativeRenderingData
	{
		public VolumeTransform volumeTransform;
	}
	class SimpleTrianglesRenderEventTypes : RenderEventTypes
	{
		public const int ACTION_RENDER = RenderEventTypes.BASE_ACTION_COUNT + 0;
	}

	#endregion

	private ActionSimpleTriangles action;

	protected SimpleTrianglesNativeRenderingData simpleTrianglesNativeRenderingData;

	#region AbstractUnityAction Overrides

	protected override AbstractRenderingAction NativeAction
	{
		get { return action; }
	}

	protected override void InitAction()
	{
		action = new();
	}

	protected override void InitAdditionalNativeStruct()
	{
		simpleTrianglesNativeRenderingData = new()
		{
			volumeTransform = new()
		};

		additionalNativeStructDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<SimpleTrianglesNativeRenderingData>());
	}

	protected override void InitRenderingCommandBuffers()
	{
		CommandBuffer cb = new();
		cb.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(SimpleTrianglesRenderEventTypes.ACTION_RENDER), renderingActionNativeRenderingDataWrapperPtr_);
		renderingCommandBuffers_.Add(new(CameraEvent.BeforeForwardOpaque, cb));
	}

	protected override void FillAdditionalNativeRenderingData()
	{
		simpleTrianglesNativeRenderingData.volumeTransform.position = volumeCube.transform.position;
		simpleTrianglesNativeRenderingData.volumeTransform.scale = volumeCube.transform.localScale;
		simpleTrianglesNativeRenderingData.volumeTransform.rotation = volumeCube.transform.rotation;

		Marshal.StructureToPtr(simpleTrianglesNativeRenderingData, additionalNativeStructDataPtr, true);
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
