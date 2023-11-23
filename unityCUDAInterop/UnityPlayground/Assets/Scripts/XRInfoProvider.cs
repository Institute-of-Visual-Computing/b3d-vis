using System.Collections;
using System.Collections.Generic;

using Unity.Collections.LowLevel.Unsafe;
using UnityEditor;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.Management;

public class XRInfoProvider : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
        /*
        Texture2DArray tex2dArr = new Texture2DArray(
            256,
            256,
            2,
            TextureFormat.RGBA32,
            false,
            false
            );

        var blub = tex2dArr.GetPixels32(0);

        for (int i = 0; i < blub.Length; i++)
        {
            blub[i] = new Color32(255, 0, 0, 255);
        }
        tex2dArr.SetPixels32(blub, 0);

        blub = tex2dArr.GetPixels32(1);

        for (int i = 0; i < blub.Length; i++)
        {
            blub[i] = new Color32(0, 255, 0, 255);
        }
        tex2dArr.SetPixels32(blub, 1);

        AssetDatabase.CreateAsset(tex2dArr, "Assets/blub.asset");
        AssetDatabase.SaveAssets();
        */
    }

    // Update is called once per frame
    void Update()
    {
    }
}
