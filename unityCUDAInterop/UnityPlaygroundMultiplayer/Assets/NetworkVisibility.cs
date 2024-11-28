using MixedReality.Toolkit;
using Unity.Netcode;
using UnityEngine;
using XRMultiplayer;

public class NetworkVisibility : NetworkBehaviour
{
	public NetworkVariableHolder variableHolder;

	public StatefulInteractable[] statefulInteractables;

	// Start is called once before the first execution of Update after the MonoBehaviour is created
	void Start()
    {
		NetworkManager.Singleton.OnClientConnectedCallback += Singleton_OnClientConnectedCallback1;
	}

	private void Singleton_OnClientConnectedCallback1(ulong obj)
	{
		if(!IsOwner)
		{
			setState(false);
		}
	}

	public override void OnGainedOwnership()
	{
		setState(true);
	}

	public override void OnLostOwnership()
	{
		setState(false);
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
