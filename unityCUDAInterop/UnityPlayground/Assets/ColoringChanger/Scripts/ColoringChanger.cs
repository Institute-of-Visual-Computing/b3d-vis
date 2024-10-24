using MixedReality.Toolkit.UX;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;


public struct ColormapInfos
{
	public string colorMapFilePath;
	public int width;
	public int height;
	public int pixelsPerMap;
	public int colorMapCount;
	public List<string> colorMapNames;
}

public class ColoringChanger : MonoBehaviour
{
    public bool useColormap = false;

    public Color colorToUse = new(0, 1, 0);

    public Material coloringMaterial;
    
    public Image uiImage;

    public Texture colormapsTexture;
	public TextAsset colormapsText;

    public Texture transferFunctiontexture;

	public GameObject buttonPrefab;
	public Transform buttonsList;
	public Material buttonsMaterial;

	RenderTexture transferRenderTex;
	Texture2D transferReadTexture;
    int colorMapCount;

    [Min(0)]
    public int selectedColorMap = 0;

    float colorMapUvHeight;

    public ComputeShader transferFunctionComputeShader;

    public ImageManipulationPositionProvider positionProvider;

    Vector2 oldPos = new Vector2(1.1f, 0);
    bool wasValidBefore = false;

	public float SelectedColorMapFloat
	{
		get { return colorMapUvHeight * selectedColorMap + colorMapUvHeight/2.0f; }
	}

	public Texture2D TransferRenderTex
	{
		get { return transferReadTexture; }
	}

	CommandBuffer cb ;
	bool cbIsOnCam = false;

	private void Start()
    {
		cb = new();
		transferRenderTex = new(transferFunctiontexture.width, transferFunctiontexture.height, 1, RenderTextureFormat.RFloat)
		{
			enableRandomWrite = true       
        };
		transferRenderTex.Create();
		Graphics.Blit(transferFunctiontexture, transferRenderTex);

		transferReadTexture = new(transferFunctiontexture.width, transferFunctiontexture.height, TextureFormat.RFloat, false, true, false)
		{
			name = "TransferReadTexture12345"
		};
		transferReadTexture.Apply();

        if(uiImage)
        {
            uiImage.material = new(coloringMaterial);
            uiImage.material.mainTexture = colormapsTexture;
            uiImage.material.SetTexture("_TransferFunctionTexture", transferRenderTex);
        }

		ColormapInfos infos = JsonUtility.FromJson<ColormapInfos>(colormapsText.text);

		colorMapUvHeight = (1.0f / infos.height) * infos.pixelsPerMap;
		colorMapCount = infos.colorMapCount;

		var prefabButton = buttonPrefab.GetComponent<PressableButton>();
		var backplateImage = buttonPrefab.GetNamedChild("Backplate").GetComponent<RawImage>();
		var frontplateText = buttonPrefab.GetNamedChild("Frontplate").GetNamedChild("Text").GetComponent<TextMeshProUGUI>();
		for (int i = 0; i < colorMapCount; i++)
		{

			var index = i;

			backplateImage.material = new Material(buttonsMaterial);
			backplateImage.color = Color.white;
			backplateImage.material.mainTexture = transferFunctiontexture;
			backplateImage.material.SetVector("_colorMapParams", new Vector4(infos.pixelsPerMap, colorMapCount - 1 - i, 1f / infos.height));

			frontplateText.text = infos.colorMapNames[i];

			var buttonGO = Instantiate(buttonPrefab, buttonsList);
			var button = buttonGO.GetComponent<PressableButton>();
			button.OnClicked.AddListener(() => { selectedColorMap = index; });
		}


		transferFunctionComputeShader.SetTexture(transferFunctionComputeShader.FindKernel("CSMain"), "Result", transferRenderTex, 0);
        transferFunctionComputeShader.SetVector("startPos", new Vector2(0, 0));
        transferFunctionComputeShader.SetVector("endPos", new Vector2(0, 0));
        transferFunctionComputeShader.SetInt("textureWidth", transferFunctiontexture.width);
        transferFunctionComputeShader.Dispatch(transferFunctionComputeShader.FindKernel("CSMain"), transferFunctiontexture.width / 8, Mathf.Max(1, transferFunctiontexture.height / 8), 1);


		cb.DispatchCompute(transferFunctionComputeShader, transferFunctionComputeShader.FindKernel("CSMain"), transferFunctiontexture.width / 8, Mathf.Max(1, transferFunctiontexture.height / 8), 1);
		cb.CopyTexture(transferRenderTex, transferReadTexture);
		Camera.main.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, cb);
	}


    // Update is called once per frame
    void Update()
    {
        Vector2 localPos;

        selectedColorMap = Mathf.Min(selectedColorMap, colorMapCount - 1);
        if (uiImage)
        {
            uiImage.material.SetFloat("_UseColorMap", useColormap ? 1 : 0);
            uiImage.material.SetColor("_SingleColor", colorToUse);
            uiImage.material.SetFloat("_ColorMapsTextureSelectionOffset", 1 - SelectedColorMapFloat);
            uiImage.material.SetFloat("_ColorMapsTextureHeight", colorMapUvHeight);
        }

        if (positionProvider.GetPosition(out localPos))
        {

            if(!wasValidBefore)
            {
                oldPos = localPos;
                wasValidBefore = true;
            }
            // Debug.Log($"old: {oldPos}, new: {localPos}");
            transferFunctionComputeShader.SetVector("startPos", oldPos);
            transferFunctionComputeShader.SetVector("endPos", localPos);
            oldPos = localPos;

			if(!cbIsOnCam)
			{
				
			}
		}
        else
        {
			if (cbIsOnCam)
			{
				
			}

			wasValidBefore = false;
            oldPos = new(1.1f, 0);
            transferFunctionComputeShader.SetVector("startPos", oldPos);
            transferFunctionComputeShader.SetVector("endPos", oldPos);
        }

    }
}
