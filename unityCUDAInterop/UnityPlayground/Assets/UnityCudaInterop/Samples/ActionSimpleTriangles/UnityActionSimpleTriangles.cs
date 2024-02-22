using System.Runtime.InteropServices;

using UnityEngine.Rendering;
using B3D.UnityCudaInterop.NativeStructs;
using B3D.UnityCudaInterop;
using UnityEngine;

public class UnityActionSimpleTriangles : AbstractUnityRenderAction
{
	#region Native structs for this action

	class SimpleTrianglesRenderEventTypes : RenderEventTypes
	{
		public const int ACTION_RENDER = RenderEventTypes.BASE_ACTION_COUNT + 0;
	}

	#endregion

	private ActionSimpleTriangles action;

	public Texture2D colorMapsTexture;

	public TextAsset colorMapsDescription;

	ColorMaps colorMaps;

	public UnityColoringMode coloringMode = UnityColoringMode.Single;


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
		cb.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(SimpleTrianglesRenderEventTypes.ACTION_RENDER), unityRenderingDataPtr);
		renderingCommandBuffers_.Add(new(CameraEvent.BeforeForwardOpaque, cb));
	}

	protected override void FillAdditionalNativeRenderingData()
	{
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
		colorMaps = ColorMaps.load(colorMapsDescription.text);
		unityRenderingData.colorMapsTexture = new(colorMapsTexture.GetNativeTexturePtr(), new((uint)colorMaps.width, (uint)colorMaps.height, 1));

		unityRenderingData.coloringInfo.coloringMode = UnityColoringMode.Single;
		unityRenderingData.coloringInfo.singleColor = new Vector4(0, 1, 0, 1);
		unityRenderingData.coloringInfo.selectedColorMap = colorMaps.firstColorMapYTextureCoordinate;
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
