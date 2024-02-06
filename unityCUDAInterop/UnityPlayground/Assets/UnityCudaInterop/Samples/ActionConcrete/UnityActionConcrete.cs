using System.Collections;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Rendering;

public class UnityActionConcrete : MonoBehaviour
{
    private ActionConcrete action;
    
    private Texture2D tex;
	CommandBuffer immediate;

	// Start is called before the first frame update
	void Start()
    {
        action = new();
        tex = new Texture2D(1024, 1024, TextureFormat.ARGB32, false);
		tex.SetPixels(0, 0, 1, 1, new Color[] { new Color() });
		tex.Apply();
		immediate = new CommandBuffer();
		StartCoroutine(WaitForTexturePointer());
    }

    void OnDestroy()
    {
        action.TeardownAction();
        action.DestroyAction();
    }

	IEnumerator WaitForTexturePointer()
	{
		yield return new WaitForEndOfFrame();
		tex.GetNativeTexturePtr();
		immediate.IssuePluginEventAndData(action.RenderEventAndDataFuncPointer, action.MapEventId(0), tex.GetNativeTexturePtr());
		Graphics.ExecuteCommandBuffer(immediate);
	}
}
