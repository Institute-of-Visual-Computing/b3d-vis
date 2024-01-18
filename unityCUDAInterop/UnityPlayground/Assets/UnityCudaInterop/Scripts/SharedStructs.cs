using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.VisualScripting;
using UnityEngine;
using static UnityEngine.Camera;
using UnityEngine.XR;

namespace B3D
{
    namespace UnityCudaInterop
    {
		struct EyeCamera
		{
			public EyeCamera(int eyeIdx, StereoscopicEye camEye, XRNode n, InputFeatureUsage<Vector3> feature)
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

		namespace NativeStructs
        {
            [StructLayout(LayoutKind.Sequential)]
            public struct TextureExtent
            {
                public uint Width;
                public uint Height;
                public uint Depth;

                public TextureExtent(uint width, uint height, uint depth)
                {
                    this.Width = width;
                    this.Height = height;
                    this.Depth = depth;
                }
            }
        
            [StructLayout(LayoutKind.Sequential)]
	        public struct NativeUnityTexture
	        {
		        public IntPtr TexturePointer;
		        public TextureExtent Extent;

                public NativeUnityTexture(IntPtr texturePointer, TextureExtent extent)
                {
                    this.TexturePointer = texturePointer;
                    this.Extent = extent;
                }
	        }
        
            [StructLayout(LayoutKind.Sequential)]
            public struct NativeCameraData
            {
                public Vector3 Origin;
                public Vector3 At;
                public Vector3 Up;
                public float CosFovY;
                public float FovY;
				public bool directionsAvailable;
				public Vector3 dir00;
				public Vector3 dirDu;
				public Vector3 dirDv;
            }
            
            [StructLayout(LayoutKind.Sequential)]
            public struct NativeTextureData
            {
                public NativeUnityTexture ColorTexture;
                public NativeUnityTexture DepthTexture;
                public static NativeTextureData CREATE()
                {
                    NativeTextureData ntd = new();
                    ntd.ColorTexture = new NativeUnityTexture();
                    ntd.DepthTexture = new NativeUnityTexture();
                    return ntd;
                }

                public NativeTextureData(NativeUnityTexture colorTexture, NativeUnityTexture depthTexture)
                {
                    this.ColorTexture = colorTexture;
                    this.DepthTexture = depthTexture;
                }
            }


			[StructLayout(LayoutKind.Sequential)]
			public struct NativeRenderingData
			{
				public int EyeCount;

				[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
				public NativeCameraData[] NativeCameradata;

				public NativeRenderingData(int eyeCount, NativeCameraData[] nativeCameradata)
				{
					this.EyeCount = eyeCount;
					NativeCameradata = new NativeCameraData[2];
					NativeCameradata[0] = nativeCameradata[0];
					NativeCameradata[1] = nativeCameradata[1];	
				}

				public static NativeRenderingData CREATE()
				{
					var nrd = new NativeRenderingData();
					nrd.EyeCount = 1;
					nrd.NativeCameradata = new NativeCameraData[2];
					return nrd;
				}
			}


			[StructLayout(LayoutKind.Sequential)]
			public struct NativeRenderingDataWrapper
			{
				public NativeRenderingData NativeRenderingData;

				public IntPtr AdditionalDataPointer;

				public NativeRenderingDataWrapper(NativeRenderingData nativeRenderingData, IntPtr additionalDataPointer)
				{
					this.NativeRenderingData = nativeRenderingData;
					this.AdditionalDataPointer = additionalDataPointer;
				}

				public static NativeRenderingDataWrapper CREATE()
				{
					NativeRenderingDataWrapper nrdw = new();
					nrdw.NativeRenderingData = NativeRenderingData.CREATE();
					nrdw.AdditionalDataPointer = IntPtr.Zero;
					return nrdw;
				}	

			}

			[StructLayout(LayoutKind.Sequential)]
			struct VolumeTransform
			{
				public Vector3 position;
				public Vector3 scale;
				public Quaternion rotation;
			};
		}
    }
}
