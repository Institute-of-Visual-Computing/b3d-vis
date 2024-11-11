using MixedReality.Toolkit.UX;
using System;
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

public class ColoringChanger : MonoBehaviour, ITransferFunctionReadTextureProvider
{
	public bool useColormap = false;
    public Color colorToUse = new(0, 1, 0);

    public Texture colormapsTexture;
	public TextAsset colormapsText;

	public GameObject buttonPrefab;
	public Transform buttonsList;
	public Material buttonsMaterial;

    int colorMapCount;

	[Min(0)]
    public int selectedColorMap = 0;

    float colorMapUvHeight;







	public ColoringChangerInteractable coloringChangerInteractable;

	public Texture initialTransferFunctionTexture;

	public MeshRenderer drawPanelMeshRenderer;

	public Material drawPanelMaterial;

	public ComputeShader drawComputeShader;

	public Texture2D TransferFunctionReadTexture
	{
		get { return transferReadTexture; }
	}

	int transferTextureWidth;
	int transferTextureHeight;

	RenderTexture transferRenderTex;
	Texture2D transferReadTexture;

	int drawComputeShaderMainKernel;
	CommandBuffer drawComputeShaderCommandBuffer;


	public float SelectedColorMapFloat
	{
		get { return colorMapUvHeight * selectedColorMap + colorMapUvHeight/2.0f; }
	}

	private void Start()
    {
		coloringChangerInteractable.StartEndPositionEvent += (start, end) =>
		{
			
			if (drawComputeShader != null)
			{
				// Start
				drawComputeShader.SetVector("startPos", start);
				drawComputeShader.SetVector("endPos", end);
			}
			else
			{
				if ((start.x < 0.0f || start.x > 1.0f) && (end.x < 0.0f || end.x > 1.0f))
				{
					return;
				}

				float m = (end.y - start.y) / (end.x - start.x);
				float c = end.y - m * end.x;

				int startPix = Math.Min(transferTextureWidth - 1, Math.Max(0, (int)(Mathf.Min(start.x, end.x) * transferTextureWidth)));
				int endPix = Math.Min(transferTextureWidth - 1, Math.Max(0, (int)(Mathf.Max(start.x, end.x) * transferTextureWidth)));

				int size = Math.Abs(startPix - endPix) + 1;

				Color[] colors = new Color[size * transferTextureHeight];
				float widthFract = 1.0f / transferTextureWidth;
				float maxVal = float.NegativeInfinity;
				for (int i = 0; i < size; i++)
				{
					colors[i].r = Mathf.Max(0.0f, Mathf.Min(1.0f, ((startPix + i) * widthFract) * m + c));
					maxVal = Mathf.Max(maxVal, colors[i].r);
				}
				if (startPix == 0)
				{
					colors[0].r = 0;
				}

				for (int i = 1; i < transferTextureHeight; i++)
				{
					Array.Copy(colors, i * size, colors, startPix, size);
				}

				transferReadTexture.SetPixels(startPix, 0, size, transferTextureHeight, colors);
				transferReadTexture.Apply();
				Graphics.CopyTexture(transferReadTexture, transferRenderTex);
			}
		};

		transferTextureWidth = initialTransferFunctionTexture != null ? initialTransferFunctionTexture.width : 512;
		transferTextureHeight = initialTransferFunctionTexture != null ? initialTransferFunctionTexture.height : 1;
		transferRenderTex = new(transferTextureWidth, transferTextureHeight, 1, RenderTextureFormat.RFloat)
		{
			enableRandomWrite = true,
			name = "TransferfunctionRenderTexture"
		};

		transferRenderTex.Create();
		if (initialTransferFunctionTexture != null)
		{
			Graphics.Blit(initialTransferFunctionTexture, transferRenderTex);
		}

		transferReadTexture = new(transferTextureWidth, transferTextureHeight, TextureFormat.RFloat, false, true, false)
		{
			name = "TransferfunctionReadTexture"
		};
		transferReadTexture.Apply();


		if (drawPanelMeshRenderer == null)
		{
			drawPanelMeshRenderer = GetComponent<MeshRenderer>();
		}

		if (drawPanelMeshRenderer != null)
		{
			if (drawPanelMaterial != null)
			{
				drawPanelMeshRenderer.material = new(drawPanelMaterial);
				drawPanelMeshRenderer.material.SetTexture("_TransferFunctionTexture", transferRenderTex);
			}
			else
			{
				Debug.Log("Draw Panel Material is not set!");
			}
		}
		else
		{
			Debug.Log("Draw Panel MeshRenderer is not set!");
		}

		if (drawComputeShader != null)
		{
			drawComputeShaderMainKernel = drawComputeShader.FindKernel("CSMain");

			drawComputeShader.SetTexture(drawComputeShaderMainKernel, "Result", transferRenderTex, 0);
			drawComputeShader.SetVector("startPos", new Vector2(0, 0));
			drawComputeShader.SetVector("endPos", new Vector2(0, 0));
			drawComputeShader.SetInt("textureWidth", transferTextureWidth);

			drawComputeShader.Dispatch(drawComputeShaderMainKernel, transferTextureWidth / 8, Mathf.Max(1, transferTextureHeight / 8), 1);

			drawComputeShaderCommandBuffer = new();

			drawComputeShaderCommandBuffer.DispatchCompute(drawComputeShader, drawComputeShaderMainKernel, transferTextureWidth / 8, Mathf.Max(1, transferTextureHeight / 8), 1);
			drawComputeShaderCommandBuffer.CopyTexture(transferRenderTex, transferReadTexture);
			Camera.main.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, drawComputeShaderCommandBuffer);
		}
		else
		{
			Debug.Log("Draw ComputeShader is not set!");
		}

		if (drawPanelMeshRenderer != null)
		{
			drawPanelMeshRenderer.material.SetTexture("_MainTex", colormapsTexture);

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
			backplateImage.material.SetVector("_colorMapParams", new Vector4(infos.pixelsPerMap, colorMapCount - 1 - i, 1f / infos.height));

			frontplateText.text = infos.colorMapNames[i];

			var buttonGO = Instantiate(buttonPrefab, buttonsList);
			var button = buttonGO.GetComponent<PressableButton>();
			button.OnClicked.AddListener(() => { selectedColorMap = index; });
		}
	}

    // Update is called once per frame
    void Update()
    {
        selectedColorMap = Mathf.Min(selectedColorMap, colorMapCount - 1);
        if (drawPanelMeshRenderer != null)
        {
			drawPanelMeshRenderer.material.SetFloat("_UseColorMap", useColormap ? 1 : 0);
			drawPanelMeshRenderer.material.SetColor("_SingleColor", colorToUse);
			drawPanelMeshRenderer.material.SetFloat("_ColorMapsTextureSelectionOffset", 1 - SelectedColorMapFloat);
			drawPanelMeshRenderer.material.SetFloat("_ColorMapsTextureHeight", colorMapUvHeight);
        }
    }
}
