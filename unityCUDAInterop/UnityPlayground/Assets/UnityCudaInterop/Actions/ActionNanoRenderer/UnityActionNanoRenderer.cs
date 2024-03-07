
using System.Runtime.InteropServices;

using System.Collections;
using B3D.UnityCudaInterop;
using B3D.UnityCudaInterop.NativeStructs;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Profiling;
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

	public Texture2D colorMapsTexture;
	Texture2D transferFunctionTexture;

	public TextAsset colorMapsDescription;

	ColorMaps colorMaps;

	public UnityColoringMode coloringMode = UnityColoringMode.Single;

	CustomSampler sampler;

	#region AbstractUnityAction Overrides


	protected override AbstractRenderingAction NativeAction
	{
		get { return action_; }
	}

	protected override void InitAction()
	{
		action_ = new();
	}

	static readonly ProfilerMarker s_nanoRendererProfileMarker = new Unity.Profiling.ProfilerMarker(Unity.Profiling.ProfilerCategory.Render, "NanoRendererOpaque");
	protected override void InitRenderingCommandBuffers()
	{
		CommandBuffer cb = new();
		
		cb.BeginSample(s_nanoRendererProfileMarker);
		cb.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(ActionNanoRendererRenderEventTypes.ACTION_RENDER), unityRenderingDataPtr);
		cb.EndSample(s_nanoRendererProfileMarker);
		renderingCommandBuffers_.Add(new(CameraEvent.BeforeForwardOpaque, cb));
	}

	protected override void FillAdditionalNativeRenderingData()
	{
		// Fill struct with custom data and copy struct to unmanaged code.
		unityRenderingData.coloringInfo.coloringMode = coloringMode;
		unityRenderingData.coloringInfo.selectedColorMap = colorMaps.firstColorMapYTextureCoordinate;

		unityRenderingData.volumeTransform.position = volumeCube.transform.position;
		unityRenderingData.volumeTransform.scale = volumeCube.transform.localScale;
		unityRenderingData.volumeTransform.rotation = volumeCube.transform.rotation;
	}

	#endregion AbstractUnityAction Overrides


	/// TODO: Current approach is to override and call parent methods like shown below. Not nice. Change to smth other
	#region Unity Methods

	protected IEnumerator StartAfter()
	{
		yield return new WaitForSeconds(10);
		colorMaps = ColorMaps.load(colorMapsDescription.text);
		sampler = CustomSampler.Create("NanoRenderSampler", true);

		base.Start();

		transferFunctionTexture = new Texture2D(512, 1, TextureFormat.RFloat, false, true, false);
		Color[] transferValues = new Color[512];
		float colStep = 1.0f / (transferValues.Length - 1);
		for (int i = 0; i < transferValues.Length; i++)
		{
			transferValues[i] = new Color(colStep * i, 0, 0);
		}
		transferFunctionTexture.SetPixels(transferValues);
		transferFunctionTexture.Apply();
		unityRenderingData.transferFunctionTexture = new(transferFunctionTexture.GetNativeTexturePtr(), new((uint)transferFunctionTexture.width, (uint)transferFunctionTexture.height, 1));


		unityRenderingData.colorMapsTexture = new(colorMapsTexture.GetNativeTexturePtr(), new((uint)colorMaps.width, (uint)colorMaps.height, 1));

		unityRenderingData.coloringInfo.coloringMode = UnityColoringMode.Single;
		unityRenderingData.coloringInfo.singleColor = new Vector4(0, 1, 0, 1);
		unityRenderingData.coloringInfo.selectedColorMap = colorMaps.firstColorMapYTextureCoordinate;
		unityRenderingData.coloringInfo.backgroundColors = new Vector4[2] { Vector4.zero, Vector4.zero };
		yield return null;
	}

	protected override void Start()
	{
		StartCoroutine(StartAfter());
		/*
		colorMaps = ColorMaps.load(colorMapsDescription.text);
		sampler = CustomSampler.Create("NanoRenderSampler", true);

		base.Start();

		transferFunctionTexture = new Texture2D(512, 1, TextureFormat.RFloat, false, true, false);
		Color[] transferValues = new Color[512];
		float colStep = 1.0f / (transferValues.Length - 1);
		for (int i = 0; i < transferValues.Length; i++)
		{
			transferValues[i] = new Color(colStep * i, 0, 0);
		}
		transferFunctionTexture.SetPixels(transferValues);
		transferFunctionTexture.Apply();
		unityRenderingData.transferFunctionTexture = new(transferFunctionTexture.GetNativeTexturePtr(), new((uint)transferFunctionTexture.width, (uint)transferFunctionTexture.height, 1));


		unityRenderingData.colorMapsTexture = new(colorMapsTexture.GetNativeTexturePtr(), new((uint)colorMaps.width, (uint)colorMaps.height, 1));

		unityRenderingData.coloringInfo.coloringMode = UnityColoringMode.Single;
		unityRenderingData.coloringInfo.singleColor = new Vector4(0, 1, 0, 1);
		unityRenderingData.coloringInfo.selectedColorMap = colorMaps.firstColorMapYTextureCoordinate;
		unityRenderingData.coloringInfo.backgroundColors = new Vector4[2] { Vector4.zero, Vector4.zero };*/
	}

	protected override void Update()
	{
		if (Input.GetKeyDown(KeyCode.T))
		{
			if(readyForUpdate_)
			{ 
				SetTextures();
			}
		}
		base.Update();
		
	}

	protected override void OnDestroy()
	{
		base.OnDestroy();
	}

	#endregion Unity Methods

}
