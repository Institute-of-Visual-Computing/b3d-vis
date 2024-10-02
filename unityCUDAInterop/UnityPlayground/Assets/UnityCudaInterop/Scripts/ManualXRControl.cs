using System.Collections;
using UnityEngine;
using System.Collections.Generic;
using UnityEngine.XR.Management;
using System;
using UnityEngine.XR;
using UnityEngine.SubsystemsImplementation;
using B3D.UnityCudaInterop.NativeStructs;
using System.Runtime.InteropServices;
using B3D.UnityCudaInterop;
public class ManualXRControl : MonoBehaviour
{


	void StartXR()
	{
		
		if (XRGeneralSettings.Instance.Manager.activeLoader != null)
		{
			XRGeneralSettings.Instance.Manager.StopSubsystems();
			XRGeneralSettings.Instance.Manager.DeinitializeLoader();
		}
		XRGeneralSettings.Instance.Manager.InitializeLoaderSync();
		XRGeneralSettings.Instance.Manager.StartSubsystems();
	}

	void StopXR()
	{
		Debug.Log("Stopping XR...");

		XRGeneralSettings.Instance.Manager.StopSubsystems();
		XRGeneralSettings.Instance.Manager.DeinitializeLoader();
		Debug.Log("XR stopped completely.");
	}

	void Start()
	{
		StartXR();
	}

	private void SubsystemManager_afterReloadSubsystems()
	{
		
	}

	private void Update()
	{
		/*	
			var displaySubSytems = new List<XRDisplaySubsystem>();
			SubsystemManager.GetInstances<XRDisplaySubsystem>(displaySubSytems);
			XRDisplaySubsystem.XRRenderPass renderpass;
			displaySubSytems[0].GetRenderPass(0, out renderpass);
			Debug.Log(renderpass.GetRenderParameterCount());
			XRDisplaySubsystem.XRRenderParameter[] renderParameter = new XRDisplaySubsystem.XRRenderParameter[2];
			renderpass.GetRenderParameter(Camera.main, 0, out renderParameter[0]);
			renderpass.GetRenderParameter(Camera.main, 1, out renderParameter[1]);
		
			var ret0 = Camera.main.projectionMatrix.MultiplyPoint(new Vector3(0, 0, -1));
			var ret = renderParameter[0].view.MultiplyPoint(new Vector3(-0.011f,0,0));

			Debug.Log("TOll");
		*/
	}

	private void OnPreRender()
	{
		
	}

	void OnDestroy()
	{
		StopXR();
	}
}
