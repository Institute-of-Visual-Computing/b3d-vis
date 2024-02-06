using B3D.UnityCudaInterop.NativeStructs;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.XR;

public class ActionTextureProvider
{
	RenderTextureDescriptor renderTextureDescriptor_;
	Texture externalTargetTexture_ = null;
	Texture oldExternalTexture_ = null;
	TextureExtent externalTargetTextureExtent_;

	public Texture ExternalTargetTexture { get { return externalTargetTexture_; } }
	public TextureExtent ExternalTargetTextureExtent { get { return externalTargetTextureExtent_; } }

	readonly RenderTextureDescriptor monoDefaultRenderTextureDescriptor = new RenderTextureDescriptor(Screen.width, Screen.height, RenderTextureFormat.ARGB32)
	{
		volumeDepth = 2,
	};

	public bool renderTextureDescriptorChanged()
	{
		List<XRDisplaySubsystemDescriptor> displays = new List<XRDisplaySubsystemDescriptor>();
		SubsystemManager.GetSubsystemDescriptors(displays);
		List<XRDisplaySubsystem> displaysSubsystems = new();
		SubsystemManager.GetSubsystems<XRDisplaySubsystem>(displaysSubsystems);

		if (XRSettings.isDeviceActive)
		{
			return !renderTextureDescriptor_.Equals(XRSettings.eyeTextureDesc);
		}
		return !renderTextureDescriptor_.Equals(monoDefaultRenderTextureDescriptor);
	}

	public Texture createExternalTargetTexture()
	{
		if (XRSettings.isDeviceActive)
		{
			renderTextureDescriptor_ = XRSettings.eyeTextureDesc;
		}
		else
		{
			renderTextureDescriptor_ = monoDefaultRenderTextureDescriptor;
		}

		oldExternalTexture_ = externalTargetTexture_;
		Texture2DArray t2dArr = new Texture2DArray(
				renderTextureDescriptor_.width,
				renderTextureDescriptor_.height,
				renderTextureDescriptor_.volumeDepth,
				renderTextureDescriptor_.graphicsFormat,
				TextureCreationFlags.None,
				0
				);
		
		t2dArr.Apply();
		externalTargetTextureExtent_ = new((uint)t2dArr.width, (uint)t2dArr.height, (uint)t2dArr.depth);
		externalTargetTexture_ = t2dArr;
		if (renderTextureDescriptor_.volumeDepth > 1)
		{
			
			// nativeRenderingData.eyeCount = 2;

		}
		else
		{
			
			// nativeRenderingData.eyeCount = 1;
		}
		
		return externalTargetTexture_;
	}
}
