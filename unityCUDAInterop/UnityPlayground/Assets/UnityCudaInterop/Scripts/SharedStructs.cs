using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.XR;

namespace B3D
{
    namespace UnityCudaInterop
    {
		public struct EyeCamera
		{
			public EyeCamera(int eyeIdx, UnityEngine.Camera.StereoscopicEye camEye, XRNode n, InputFeatureUsage<Vector3> feature)
			{
				eyeIndex = eyeIdx;
				cameraEye = camEye;
				xrNode = n;
				nodeUsage = feature;
			}
			public readonly int eyeIndex;
			public readonly Camera.StereoscopicEye cameraEye;
			public readonly XRNode xrNode;
			public readonly InputFeatureUsage<Vector3> nodeUsage;
		}

		class SharedMembers
		{
			public static readonly EyeCamera[] eyeCameraMapping = new EyeCamera[]
			{
				new( 0, Camera.StereoscopicEye.Left, XRNode.LeftEye, CommonUsages.leftEyePosition),
				new( 1, Camera.StereoscopicEye.Right, XRNode.RightHand, CommonUsages.rightEyePosition)
			};
		}

		[Serializable]
		public class ColorMaps
		{
			public string colorMapFilePath;
			public List<string> colorMapNames;
			public int height;
			public int width;
			public int pixelsPerMap;
			public int colorMapCount;

			[NonSerialized]
			public float firstColorMapYTextureCoordinate;
			[NonSerialized]
			public float colorMapHeightNormalized;

			public static ColorMaps load(string jsonString)
			{
				ColorMaps cms = JsonUtility.FromJson<ColorMaps>(jsonString);
				cms.colorMapHeightNormalized = (1.0f / (float)cms.height) * cms.pixelsPerMap;
				cms.firstColorMapYTextureCoordinate = cms.colorMapHeightNormalized / 2.0f;
				return cms;
			}
		}

		public class RenderEventTypes
		{
			public const int MAX_EVENT_COUNT = 10;

			public const int ACTION_INITIALIZE = 0;
			public const int ACTION_SET_TEXTURES = 1;

			public const int BASE_ACTION_COUNT = ACTION_SET_TEXTURES + 1;

		}

		namespace NativeStructs
        {
			// defined in Action.h - using UnityCamera = renderer::Camera;
			[StructLayout(LayoutKind.Sequential)]
			public struct UnityCamera
			{
				public Vector3 Origin;
				public Vector3 At;
				public Vector3 Up;
				public float CosFovY;
				public float FovY; // in radians
				public bool directionsAvailable;
				public Vector3 dir00;
				public Vector3 dirDu;
				public Vector3 dirDv;
			};

			// defined in Action.h - using UnityExtent = renderer::Extent;
			[StructLayout(LayoutKind.Sequential)]
            public struct UnityExtent
			{
                public uint Width;
                public uint Height;
                public uint Depth;

                public UnityExtent(uint width, uint height, uint depth)
                {
                    this.Width = width;
                    this.Height = height;
                    this.Depth = depth;
                }
            }

			// defined in Action.h - using UnityRenderMode = renderer::RenderMode;
			public enum UnityRenderMode : int
			{
				mono = 0,
				stereo
			};

			// defined in Action.h - using UnityView = renderer::View;
			[StructLayout(LayoutKind.Sequential)]
			public struct UnityView
			{
				[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
				public UnityCamera[] UnityCameras;

				public UnityRenderMode mode;

				public static UnityView CREATE()
				{
					UnityView uv = new();
					uv.UnityCameras = new UnityCamera[2];
					uv.mode = UnityRenderMode.mono;
					return uv;
				}

			};

			// defined in Action.h - using UnityColoringMode = renderer::ColoringMode;
			public enum UnityColoringMode : int
			{
				Single = 0,
				Colormap = 1
			};

			// defined in Action.h - using UnityColoringInfo = renderer::ColoringInfo;
			[StructLayout(LayoutKind.Sequential)]
			public struct UnityColoringInfo
			{
				public UnityColoringMode coloringMode;
				public Vector4 singleColor;
				public float selectedColorMap;
			};

			// defined in Action.h - struct UnityTexture
			[StructLayout(LayoutKind.Sequential)]
			public struct UnityTexture
			{
				public IntPtr TexturePointer;
				public UnityExtent Extent;

				public UnityTexture(IntPtr texturePointer, UnityExtent extent)
				{
					this.TexturePointer = texturePointer;
					this.Extent = extent;
				}
			}

			// defined in Action.h - struct UnityRenderTargets
			[StructLayout(LayoutKind.Sequential)]
			public struct UnityRenderTargets
			{
				public UnityTexture colorRt;
				public UnityTexture minMaxRt;
			};

			// defined in Action.h - struct UnityVolumeTransform
			[StructLayout(LayoutKind.Sequential)]
			public struct UnityVolumeTransform
			{
				public Vector3 position;
				public Vector3 scale;
				public Quaternion rotation;
			};

			// defined in Action.h - struct UnityRenderingData
			[StructLayout(LayoutKind.Sequential)]
			public struct UnityRenderingData
			{
				public UnityRenderTargets renderTargets;
				public UnityView view;
				public UnityVolumeTransform volumeTransform;
				public UnityTexture colorMapsTexture;
				public UnityColoringInfo coloringInfo;

			}
		}
    }
}
