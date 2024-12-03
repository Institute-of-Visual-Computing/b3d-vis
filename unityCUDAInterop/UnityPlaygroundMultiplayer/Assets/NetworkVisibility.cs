using MixedReality.Toolkit;
using Unity.Netcode;
using UnityEngine;
using XRMultiplayer;

public class NetworkVisibility : NetworkBehaviour
{
	public StatefulInteractable[] statefulInteractables;

	// Start is called once before the first execution of Update after the MonoBehaviour is created
	void Start()
    {
		NetworkManager.Singleton.OnClientConnectedCallback += Singleton_OnClientConnectedCallback1;
	}

	private void Singleton_OnClientConnectedCallback1(ulong obj)
	{
		if(!IsOwner && NetworkManager.LocalClientId != NetworkManager.ServerClientId)
		{
			setState(false);
		}
	}

	public override void OnGainedOwnership()
	{
		if (NetworkManager.LocalClientId != NetworkManager.ServerClientId)
		{
			Debug.LogWarning("Gained as client");
			// Client but not server
			setState(true);
		}
		else
		{
			// Server
			if (OwnerClientId == NetworkManager.ServerClientId)
			{
				Debug.LogWarning("Gained as Server");
				// Give me ownership only if i'm the owner :D
				setState(true);
			}
		}
	}

	public override void OnLostOwnership()
	{
		if (NetworkManager.LocalClientId != NetworkManager.ServerClientId)
		{
			Debug.LogWarning("Lost as client");
			// Client but not server
			setState(false);
		}
		else
		{
			// Server
			if (OwnerClientId != NetworkManager.ServerClientId)
			{
				Debug.LogWarning("Lost as server");
				// Disable only if server loses ownershio
				setState(false);
			}
		}
	}

	void setState(bool state)
	{
		if (statefulInteractables != null)
		{
			foreach (var item in statefulInteractables)
			{
				item.ForceSetToggled(false);
				item.enabled = state;
			}
		}
	}
}
