using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using static UnityEngine.Camera;
using UnityEngine.XR;
using B3D.UnityCudaInterop.NativeStructs;

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

			protected UnityRenderingData unityRenderingData;

			/// <summary>
			/// Pointer to unmanaged memory for custom data. Gets destroyed automatically.
			/// </summary>
			protected System.IntPtr unityRenderingDataPtr;
			
			protected CommandBuffer commandBuffer;

			protected List<Tuple<CameraEvent, CommandBuffer>> renderingCommandBuffers_;

			protected bool readyForUpdate_ = false;

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
				if(readyForUpdate_)
				{ 
					if (textureProvider_.renderTextureDescriptorChanged())
					{
						SetTextures();
					}
					else
					{
						FillNativeRenderingData();

						Marshal.StructureToPtr(unityRenderingData, unityRenderingDataPtr, true);
					}
				}
			}

			protected virtual void OnDestroy()
			{
				readyForUpdate_ = false;
				RemoveRenderingCommandBuffersFromCamera();

				NativeAction.TeardownAction();
				NativeAction.DestroyAction();

				Marshal.FreeHGlobal(unityRenderingDataPtr);
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

				Marshal.StructureToPtr(unityRenderingData, unityRenderingDataPtr, true);

				CommandBuffer immediate = new();
				immediate.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(RenderEventTypes.ACTION_INITIALIZE), unityRenderingDataPtr);
				Graphics.ExecuteCommandBuffer(immediate);
				yield return new WaitForEndOfFrame();
				yield return new WaitForEndOfFrame();
				AddRenderingCommandBuffersToCamera();
				readyForUpdate_ = true;
			}

			protected virtual IEnumerator WaitEndOfFrameAfterImmediateCommandBufferExec()
			{
				yield return new WaitForEndOfFrame();
				// yield return new WaitForSeconds(1);
				AddRenderingCommandBuffersToCamera();
				readyForUpdate_ = true;
			}

			protected virtual void InitAllObjects()
			{
				volumeObjectMaterial = new(objectMaterial);
				objectRenderer.material = volumeObjectMaterial;

				renderingCommandBuffers_ = new();
				textureProvider_ = new();

				unityRenderingData = new();
				unityRenderingData.view = UnityView.CREATE();
				unityRenderingDataPtr = Marshal.AllocHGlobal(Marshal.SizeOf<UnityRenderingData>());

				InitAction();
				
				InitRenderingCommandBuffers();
			}

			protected virtual void SetTextures(bool init = false)
			{
				textureProvider_.createExternalTargetTexture();

				// quadFullscreenMaterial.SetTexture("_MainTex", action.TextureProvider.ExternalTargetTexture);
				volumeObjectMaterial.SetTexture("_MainTex", textureProvider_.ExternalTargetTexture);


				unityRenderingData.renderTargets.minMaxRt.Extent.Depth = 0;
				unityRenderingData.renderTargets.minMaxRt.TexturePointer = IntPtr.Zero;


				unityRenderingData.renderTargets.colorRt.TexturePointer = textureProvider_.ExternalTargetTexture.GetNativeTexturePtr();
				unityRenderingData.renderTargets.colorRt.Extent = textureProvider_.ExternalTargetTextureExtent;

				// Execute only if we're updating the texture. 
				if(!init) {

					readyForUpdate_ = false;
					RemoveRenderingCommandBuffersFromCamera();

					Marshal.StructureToPtr(unityRenderingData, unityRenderingDataPtr, true);

					CommandBuffer cbImmediate = new();
					cbImmediate.IssuePluginEventAndData(NativeAction.RenderEventAndDataFuncPointer, NativeAction.MapEventId(RenderEventTypes.ACTION_SET_TEXTURES), unityRenderingDataPtr);
					Graphics.ExecuteCommandBuffer(cbImmediate);
					StartCoroutine(WaitEndOfFrameAfterImmediateCommandBufferExec());
				}
			}

			protected virtual void fillNativeRenderingDataWrapper()
			{
				if (XRSettings.isDeviceActive)
				{
					unityRenderingData.view.mode = UnityRenderMode.stereo;

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

						
						unityRenderingData.view.UnityCameras[nodeUsage.eyeIndex].dir00 = camLowerLeft;
						unityRenderingData.view.UnityCameras[nodeUsage.eyeIndex].dirDu = onePxDirectionU;
						unityRenderingData.view.UnityCameras[nodeUsage.eyeIndex].dirDv = onePxDirectionV;
						unityRenderingData.view.UnityCameras[nodeUsage.eyeIndex].directionsAvailable = true;
					}
				}
				else
				{
					unityRenderingData.view.mode = UnityRenderMode.mono;
					SetNativeRenderingCameraData(targetCamera.transform.position, targetCamera.transform.forward, targetCamera.transform.up, targetCamera.fieldOfView, 0);

					Vector3 cameraWorldPosition = targetCamera.transform.position;
					var eyePos = cameraWorldPosition;

					var upperLeft = targetCamera.ScreenToWorldPoint(new Vector3(0, textureProvider_.ExternalTargetTextureExtent.Height - 1, 1));
					var upperRight = targetCamera.ScreenToWorldPoint(new Vector3(textureProvider_.ExternalTargetTextureExtent.Width - 1, textureProvider_.ExternalTargetTextureExtent.Height - 1, 1));
					var lowerLeft = targetCamera.ScreenToWorldPoint(new Vector3(0, 0, 1));

					var onePxDirectionU = (upperRight - upperLeft); // / action.TextureProvider.ExternalTargetTextureExtent.Width;
					var onePxDirectionV = (upperLeft - lowerLeft); //  / action.TextureProvider.ExternalTargetTextureExtent.Height;
					var camLowerLeft = (lowerLeft - eyePos);


					unityRenderingData.view.UnityCameras[0].dir00 = camLowerLeft;
					unityRenderingData.view.UnityCameras[0].dirDu = onePxDirectionU;
					unityRenderingData.view.UnityCameras[0].dirDv = onePxDirectionV;
					unityRenderingData.view.UnityCameras[0].directionsAvailable = true;
				}
			}

			protected virtual void SetNativeRenderingCameraData(Vector3 origin, Vector3 at, Vector3 up, float fovYDegree, int eyeIndex)
			{
				UnityCamera nativeCameraData = new()
				{
					Origin = origin,
					At = at,
					Up = up,
					CosFovY = Mathf.Cos(Mathf.Deg2Rad * fovYDegree),
					FovY = Mathf.Deg2Rad * fovYDegree,
					directionsAvailable = true
				};
				unityRenderingData.view.UnityCameras[eyeIndex] = nativeCameraData;
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
				if (targetCamera)
				{
					foreach (var (evt, cb) in renderingCommandBuffers_)
					{
						targetCamera.RemoveCommandBuffer(evt, cb);
					}
				}
			}

			#endregion helpers

		}
	}
}
