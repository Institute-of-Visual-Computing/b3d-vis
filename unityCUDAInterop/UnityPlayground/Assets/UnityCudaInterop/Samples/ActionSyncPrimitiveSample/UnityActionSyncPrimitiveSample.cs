using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;

using UnityEngine.Rendering;
using System;
using UnityEngine.UI;
using static System.Net.Mime.MediaTypeNames;

public class UnityActionSyncPrimitiveSample : MonoBehaviour
{
    private ActionSyncPrimitiveSample action;

    public Texture2D tex;

    CommandBuffer cb;

	CommandBuffer immediate;
	public RawImage ri;

    // Start is called before the first frame update
    void Start()
    {
        action = new();
        tex = new Texture2D(1024, 1024, TextureFormat.ARGB32, false)
		{
			name = "LALALAL"
			
		};
		tex.SetPixels(0, 0, 1, 1, new Color[] { new Color() });
		tex.Apply();
		ri.texture = tex;

		cb = new CommandBuffer();
		tex.GetNativeTexturePtr();
		immediate = new CommandBuffer();
		// action.initializeAction(tex.GetNativeTexturePtr());

		cb.IssuePluginEventAndData(action.RenderEventAndDataFuncPointer, action.mapEventId(1), IntPtr.Zero);
		Camera.main.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, cb);
		StartCoroutine(WaitForTexturePointer());
    }

	private void Update()
	{

	}

	IEnumerator WaitForTexturePointer()
	{
		yield return new WaitForEndOfFrame();
		tex.GetNativeTexturePtr();
		Debug.Log("lalala");
		immediate.IssuePluginEventAndData(action.RenderEventAndDataFuncPointer, action.mapEventId(0), tex.GetNativeTexturePtr());
		Graphics.ExecuteCommandBuffer(immediate);
	}
}
