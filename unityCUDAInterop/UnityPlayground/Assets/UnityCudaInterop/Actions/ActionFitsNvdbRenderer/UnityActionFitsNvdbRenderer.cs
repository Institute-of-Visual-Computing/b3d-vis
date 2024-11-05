using B3D.UnityCudaInterop.NativeStructs;
using B3D.UnityCudaInterop;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

using System.Collections;
using System.Runtime.InteropServices;
using Unity.XR.CoreUtils.Datums;
using System;
using UnityEditor;
using System.Threading.Tasks;

public class UnityActionFitsNvdbRenderer : AbstractUnityRenderAction
{
	#region Native structs for this action

	class ActionFitsNvdbRendererRenderEventTypes : RenderEventTypes
	{
		public const int ACTION_RENDER = RenderEventTypes.BASE_ACTION_COUNT + 0;
	}

	#endregion

	private ActionFitsNvdbRenderer action_;


	public TextAsset colorMapsDescription;

	ColorMaps colorMaps;

	public UnityColoringMode coloringMode = UnityColoringMode.Single;

	CustomSampler sampler;

	Texture2D testColorTex;

	public ColoringChanger coloringChanger;

	public Texture colormapsTexture;
	public ServerFileCache serverFileCache;

	IntPtr fitsNvdbDataPointer;
	bool newDataAvailable = false;
	string currentVolumePath = "";
	string currentVolumeUUID = "";
	B3D.Project currentProject = null;
	byte[] fitsNvdbData;

	#region AbstractUnityAction Overrides


	protected override AbstractRenderingAction NativeAction
	{
		get { return action_; }
	}

	protected override void InitAction()
	{
		action_ = new();
	}

	static readonly ProfilerMarker s_nanoRendererProfileMarker = new Unity.Profiling.ProfilerMarker(Unity.Profiling.ProfilerCategory.Render, "FitsNvdbRendererOpaque");
	protected override void InitRenderingCommandBuffers()
	{
		CommandBuffer cb = new();

		cb.BeginSample(s_nanoRendererProfileMarker);
		cb.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(ActionFitsNvdbRendererRenderEventTypes.ACTION_RENDER), unityRenderingDataPtr);
		cb.EndSample(s_nanoRendererProfileMarker);
		renderingCommandBuffers_.Add(new(CameraEvent.BeforeForwardOpaque, cb));
	}

	protected override void FillAdditionalNativeRenderingData()
	{
		// Fill struct with custom data and copy struct to unmanaged code.
		unityRenderingData.coloringInfo.coloringMode = coloringChanger.useColormap ? UnityColoringMode.Colormap : UnityColoringMode.Single;
		unityRenderingData.coloringInfo.singleColor = coloringChanger.colorToUse;
		unityRenderingData.coloringInfo.selectedColorMap = 1-coloringChanger.SelectedColorMapFloat;
		unityRenderingData.coloringInfo.backgroundColors = new Vector4[2] { Vector4.zero, Vector4.zero };

		unityRenderingData.volumeTransform.position = volumeCube.transform.position;
		unityRenderingData.volumeTransform.scale = volumeCube.transform.localScale;
		unityRenderingData.volumeTransform.rotation = volumeCube.transform.rotation;


		unityRenderingData.nanovdbData = new();
		if(newDataAvailable)
		{
			unityRenderingData.nanovdbData.newVolumeAvailable = true;
			unityRenderingData.nanovdbData.fitsDimensions = new Vector3(
				currentProject.fitsOriginProperties.axisDimensions[0],
				currentProject.fitsOriginProperties.axisDimensions[1],
				currentProject.fitsOriginProperties.axisDimensions[2]
				);
			unityRenderingData.nanovdbData.nanoVdbFilePath = currentVolumePath;
			unityRenderingData.nanovdbData.pathStringLength = currentVolumePath.Length;
			unityRenderingData.nanovdbData.nanoVdbUUID = currentVolumeUUID;
			unityRenderingData.nanovdbData.uuidStringLength = currentVolumeUUID.Length;

			objectRenderer.enabled = true;
			objectRenderer.transform.Find("Coordinates").gameObject.SetActive(true);

			newDataAvailable = false;
		}
		else
		{
			unityRenderingData.nanovdbData.newVolumeAvailable = false;
			unityRenderingData.nanovdbData.fitsDimensions = Vector3.one;
			unityRenderingData.nanovdbData.nanoVdbFilePath = "Empty";
			unityRenderingData.nanovdbData.pathStringLength = 5;
			unityRenderingData.nanovdbData.nanoVdbUUID = "Empty";
			unityRenderingData.nanovdbData.uuidStringLength = 5;
			
		}
	}

	#endregion AbstractUnityAction Overrides

	/// TODO: Current approach is to override and call parent methods like shown below. Not nice. Change to smth other
	#region Unity Methods

	protected IEnumerator StartAfter()
	{
		yield return new WaitForSeconds(1);
		colorMaps = ColorMaps.load(colorMapsDescription.text);
		sampler = CustomSampler.Create("FitsNvdbRenderSampler", true);

		base.Start();

		unityRenderingData.transferFunctionTexture = new(coloringChanger.TransferFunctionReadTexture.GetNativeTexturePtr(), new((uint)coloringChanger.TransferFunctionReadTexture.width, (uint)coloringChanger.TransferFunctionReadTexture.height, 1));

		unityRenderingData.colorMapsTexture = new(colormapsTexture.GetNativeTexturePtr(), new((uint)colormapsTexture.width, (uint)colormapsTexture.height, 1));
		//unityRenderingData.colorMapsTexture = new(testColorTex.GetNativeTexturePtr(), new((uint)testColorTex.width, (uint)testColorTex.height, 1));

		unityRenderingData.coloringInfo.coloringMode = coloringChanger.useColormap ? UnityColoringMode.Colormap : UnityColoringMode.Single;
		unityRenderingData.coloringInfo.singleColor = coloringChanger.colorToUse;
		unityRenderingData.coloringInfo.selectedColorMap = coloringChanger.SelectedColorMapFloat;
		unityRenderingData.coloringInfo.backgroundColors = new Vector4[2] { Vector4.zero, Vector4.zero };
		yield return null;
	}

	protected override void Start()
	{
		testColorTex = new(1024, 512, TextureFormat.RGBA32, false, false);
		Color[] colors = new Color[1024 * 512];
		for (int i = 0; i < colors.Length; i++)
		{
			colors[i] = Color.red;
		}
		testColorTex.SetPixels(colors);
		testColorTex.Apply();

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
			if (readyForUpdate_)
			{
				SetTextures();
			}
		}
		base.Update();
	}

	protected override void OnDestroy()
	{
		if(fitsNvdbDataPointer != IntPtr.Zero)
		{ 
			Marshal.FreeHGlobal(fitsNvdbDataPointer);
			fitsNvdbDataPointer = IntPtr.Zero;
		}

		base.OnDestroy();
	}

	#endregion Unity Methods

	IEnumerator LoadVolume(B3D.Project project, string fileUuid)
	{
		var fileTaskTuple = serverFileCache.downloadFile(fileUuid);
		yield return new WaitUntil(() => fileTaskTuple.IsCompleted);

		if(fileTaskTuple.IsCompletedSuccessfully)
		{
			currentVolumeUUID = fileTaskTuple.Result.Item1;	
			currentVolumePath = fileTaskTuple.Result.Item2;
			currentProject = project;
			newDataAvailable = true;
		}
	}
	public void showVolume(B3D.Project project, string fileUuid)
	{
		if(!serverFileCache)
		{
			Debug.Log("No server file cache available");
			return;
		}
		StartCoroutine(LoadVolume(project, fileUuid));
	}
}

