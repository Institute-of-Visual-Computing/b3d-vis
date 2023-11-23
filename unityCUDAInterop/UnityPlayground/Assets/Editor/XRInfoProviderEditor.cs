using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.TerrainTools;
using UnityEngine;
using UnityEngine.XR;

[CustomEditor(typeof(XRInfoProvider))]
public class XRInfoProviderEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        EditorGUILayout.LabelField("DeviceName", XRSettings.loadedDeviceName);
        EditorGUILayout.LabelField("XR enabled: ", XRSettings.isDeviceActive.ToString());
        EditorGUILayout.LabelField("Texture Dimension Type: ", XRSettings.deviceEyeTextureDimension.ToString());
        EditorGUILayout.LabelField("Dimensions", XRSettings.eyeTextureWidth + " x " + XRSettings.eyeTextureHeight);
        EditorGUILayout.LabelField("TextureUsage", XRSettings.eyeTextureDesc.vrUsage.ToString());
        EditorGUILayout.LabelField("ColorFormat", XRSettings.eyeTextureDesc.colorFormat.ToString());
        EditorGUILayout.LabelField("GraphicsFormat", XRSettings.eyeTextureDesc.graphicsFormat.ToString());
        EditorGUILayout.LabelField("stereoRenderingMode", XRSettings.stereoRenderingMode.ToString());
    }
}
