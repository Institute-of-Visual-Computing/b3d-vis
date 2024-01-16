using System.Collections;
using UnityEngine;
using System.Collections.Generic;
using UnityEngine.XR.Management;
using System;
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

	void OnDestroy()
	{
		StopXR();
	}
}
