using B3D.UnityCudaInterop.NativeStructs;
using B3D.UnityCudaInterop.NativeStructs.RenderAction;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using static UnityEngine.Camera;
using UnityEngine.XR;

namespace B3D
{
	namespace UnityCudaInterop
	{
		public abstract class AbstractUnityRenderAction : MonoBehaviour
		{
			#region Members

			/// <summary>
			/// Camera which renders cuda content.
			/// </summary>
			public Camera targetCamera;

			/// <summary>
			/// GameObject with MeshRenderer and default cube mesh, where the rendered content is projected on.
			/// </summary>
			public GameObject volumeCube;

			/// <summary>
			/// Renderer attached to <see cref="volumeCube"/>
			/// </summary>
			public Renderer objectRenderer;

			/// <summary>
			/// Material for projection.
			/// </summary>
			public Material objectMaterial;

			/// <summary>
			/// Instance of <see cref="objectMaterial"/> which is applied to <see cref="objectRenderer"/>
			/// </summary>
			protected Material volumeObjectMaterial;

			/// <summary>
			/// Returns textures. TODO: Replace with Unity object and follow a component based solution.
			/// </summary>
			protected ActionTextureProvider textureProvider_;

			protected RenderingActionNativeInitData nativeInitData;
			protected System.IntPtr nativeInitDataPtr;

			protected NativeTextureData nativeTextureData;
			protected IntPtr nativeTextureDataPtr;

			protected RenderingActionNativeRenderingDataWrapper renderingActionNativeRenderingDataWrapper_;
			protected System.IntPtr renderingActionNativeRenderingDataWrapperPtr_;

			/// <summary>
			/// Pointer to unmanaged memory for custom data. Gets destroyed automatically. Derived classes must create memory in order to use custom data.
			/// </summary>
			protected System.IntPtr additionalNativeStructDataPtr;
			
			protected CommandBuffer commandBuffer;

			protected List<Tuple<CameraEvent, CommandBuffer>> renderingCommandBuffers_;

			#endregion Members

			#region Unity Methods

			protected virtual void Start()
			{
				InitAllObjects();

				SetTextures(init: true);
				FillNativeRenderingData();

				StartCoroutine(InitPluginAtEndOfFrame());
			}

			protected virtual void Update()
			{
				if (textureProvider_.renderTextureDescriptorChanged())
				{
					SetTextures();
				}
				else
				{
					FillNativeRenderingData();
				}
			}

			protected virtual void OnDestroy()
			{
				RemoveRenderingCommandBuffersFromCamera();

				NativeAction.TeardownAction();
				NativeAction.DestroyAction();


				Marshal.FreeHGlobal(additionalNativeStructDataPtr);
				Marshal.FreeHGlobal(renderingActionNativeRenderingDataWrapperPtr_);
				Marshal.FreeHGlobal(nativeInitDataPtr);
				Marshal.FreeHGlobal(nativeTextureDataPtr);
			}

			#endregion Unity Methods

			#region Abstract

			/// <summary>
			/// Provides a way to pass the concrete RenderingAction from the inherited class to its parent. Derived classes from AbstractUnityRenderAction must return their concrete RenderingAction!
			/// </summary>
			protected abstract AbstractRenderingAction NativeAction
			{
				get;
			}

			/// <summary>
			/// Creates the action object in the derived class.
			/// </summary>
			protected abstract void InitAction();

			/// <summary>
			/// If the derived class uses additional data the corresponding structs should be created here.
			/// </summary>
			protected abstract void InitAdditionalNativeStruct();

			/// <summary>
			/// Create commandbuffers for rendering purposes in this method. Pass them to <see cref="renderingCommandBuffers_"/>
			/// </summary>
			protected abstract void InitRenderingCommandBuffers();

			/// <summary>
			/// Fill the struct with custom data for rendering with data. Gets called every frame after <see cref="fillNativeRenderingDataWrapper"/> gets called.
			/// </summary>
			protected abstract void FillAdditionalNativeRenderingData();

			#endregion Abstract

			protected virtual IEnumerator InitPluginAtEndOfFrame()
			{
				yield return new WaitForEndOfFrame();

				nativeInitData = new();
				nativeInitData.textureData = nativeTextureData;
				Marshal.StructureToPtr(nativeInitData, nativeInitDataPtr, true);

				CommandBuffer immediate = new();
				immediate.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(RenderEventTypes.ACTION_INITIALIZE), nativeInitDataPtr);
				Graphics.ExecuteCommandBuffer(immediate);
				yield return new WaitForEndOfFrame();
				yield return new WaitForEndOfFrame();
				AddRenderingCommandBuffersToCamera();
			}

			protected virtual IEnumerator WaitEndOfFrameAfterImmediateCommandBufferExec()
			{
				yield return new WaitForEndOfFrame();
				// yield return new WaitForSeconds(1);
				AddRenderingCommandBuffersToCamera();
			}

			protected virtual void InitAllObjects()
			{
				volumeObjectMaterial = new(objectMaterial);
				objectRenderer.material = volumeObjectMaterial;

				renderingCommandBuffers_ = new();
				textureProvider_ = new();
				InitAction();

				renderingActionNativeRenderingDataWrapper_ = RenderingActionNativeRenderingDataWrapper.CREATE();
				renderingActionNativeRenderingDataWrapperPtr_ = Marshal.AllocHGlobal(Marshal.SizeOf<RenderingActionNativeRenderingDataWrapper>());
				InitAdditionalNativeStruct();
				renderingActionNativeRenderingDataWrapper_.AdditionalDataPointer = additionalNativeStructDataPtr;

				nativeInitData = new();
				nativeTextureData = new();

				nativeInitDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<RenderingActionNativeInitData>());
				nativeTextureDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<NativeTextureData>());

				InitRenderingCommandBuffers();
			}

			protected virtual void SetTextures(bool init = false)
			{
				nativeTextureData = new();
				textureProvider_.createExternalTargetTexture();

				// quadFullscreenMaterial.SetTexture("_MainTex", action.TextureProvider.ExternalTargetTexture);
				volumeObjectMaterial.SetTexture("_MainTex", textureProvider_.ExternalTargetTexture);

				nativeTextureData.DepthTexture.Extent.Depth = 0;
				nativeTextureData.DepthTexture.TexturePointer = IntPtr.Zero;

				nativeTextureData.ColorTexture.TexturePointer = textureProvider_.ExternalTargetTexture.GetNativeTexturePtr();
				nativeTextureData.ColorTexture.Extent = textureProvider_.ExternalTargetTextureExtent;

				Marshal.StructureToPtr(nativeTextureData, nativeTextureDataPtr, true);

				// Execute only if we're updating the texture.
				if(!init) {
					RemoveRenderingCommandBuffersFromCamera();
					
					CommandBuffer cbImmediate = new();
					cbImmediate.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(RenderEventTypes.ACTION_SET_TEXTURES), nativeTextureDataPtr);
					Graphics.ExecuteCommandBuffer(cbImmediate);
					StartCoroutine(WaitEndOfFrameAfterImmediateCommandBufferExec());
				}
			}

			protected virtual void fillNativeRenderingDataWrapper()
			{
				if (XRSettings.isDeviceActive)
				{
					renderingActionNativeRenderingDataWrapper_.NativeRenderingData.EyeCount = 2;

					Vector3 cameraWorldPosition = targetCamera.transform.position;

					XRDisplaySubsystem.XRRenderParameter[] renderParameter = new XRDisplaySubsystem.XRRenderParameter[2];

					foreach (var nodeUsage in SharedMembers.eyeCameraMapping)
					{
						// cameraWorldPosition + (nodeUsage.cameraEye == StereoscopicEye.Left ? -1.0f : 1.0f) * 0.5f * targetCamera_.stereoSeparation * targetCamera_.transform.right;
						var eyePos = cameraWorldPosition + (nodeUsage.cameraEye == StereoscopicEye.Left ? -0.5f : 0.5f) * targetCamera.stereoSeparation * targetCamera.transform.right;

						SetNativeRenderingCameraData(eyePos, targetCamera.transform.forward + eyePos, targetCamera.transform.up, targetCamera.fieldOfView, nodeUsage.eyeIndex);

						var upperLeft = targetCamera.ScreenToWorldPoint(new Vector3(0, textureProvider_.ExternalTargetTextureExtent.Height - 1, 1), (MonoOrStereoscopicEye)nodeUsage.cameraEye);
						var upperRight = targetCamera.ScreenToWorldPoint(new Vector3(textureProvider_.ExternalTargetTextureExtent.Width - 1, textureProvider_.ExternalTargetTextureExtent.Height - 1, 1), (MonoOrStereoscopicEye)nodeUsage.cameraEye);
						var lowerLeft = targetCamera.ScreenToWorldPoint(new Vector3(0, 0, 1), (MonoOrStereoscopicEye)nodeUsage.cameraEye);

						var onePxDirectionU = (upperRight - upperLeft); // / action.TextureProvider.ExternalTargetTextureExtent.Width;
						var onePxDirectionV = (upperLeft - lowerLeft); //  / action.TextureProvider.ExternalTargetTextureExtent.Height;
						var camLowerLeft = (lowerLeft - eyePos);

						renderingActionNativeRenderingDataWrapper_.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].dir00 = camLowerLeft;
						renderingActionNativeRenderingDataWrapper_.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].dirDu = onePxDirectionU;
						renderingActionNativeRenderingDataWrapper_.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].dirDv = onePxDirectionV;
						renderingActionNativeRenderingDataWrapper_.NativeRenderingData.NativeCameradata[nodeUsage.eyeIndex].directionsAvailable = true;
					}
				}
				else
				{
					renderingActionNativeRenderingDataWrapper_.NativeRenderingData.EyeCount = 1;
					SetNativeRenderingCameraData(targetCamera.transform.position, targetCamera.transform.forward, targetCamera.transform.up, targetCamera.fieldOfView, 0);
				}
				Marshal.StructureToPtr(renderingActionNativeRenderingDataWrapper_, renderingActionNativeRenderingDataWrapperPtr_, true);
			}

			protected virtual void SetNativeRenderingCameraData(Vector3 origin, Vector3 at, Vector3 up, float fovYDegree, int eyeIndex)
			{
				NativeCameraData nativeCameraData = new()
				{
					Origin = origin,
					At = at,
					Up = up,
					CosFovY = Mathf.Cos(Mathf.Deg2Rad * fovYDegree),
					FovY = Mathf.Deg2Rad * fovYDegree,
					directionsAvailable = false
				};
				renderingActionNativeRenderingDataWrapper_.NativeRenderingData.NativeCameradata[eyeIndex] = nativeCameraData;
			}

			protected virtual void FillNativeRenderingData()
			{
				fillNativeRenderingDataWrapper();
				FillAdditionalNativeRenderingData();
			}

			#region helpers
			protected virtual void AddRenderingCommandBuffersToCamera()
			{
				foreach (var (evt, cb) in renderingCommandBuffers_)
				{
					targetCamera.AddCommandBuffer(evt, cb);
				}
			}

			protected virtual void RemoveRenderingCommandBuffersFromCamera()
			{
				foreach (var (evt, cb) in renderingCommandBuffers_)
				{
					targetCamera.AddCommandBuffer(evt, cb);
				}
			}

			#endregion helpers

		}
	}
}
