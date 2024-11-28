using System;
using System.Collections;
using Unity.Netcode;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit.Locomotion.Teleportation;
using XRMultiplayer;

public class RandomTeleporter : MonoBehaviour
{
	public ClientSeatAllocator seatAllocator;
    public TeleportationAnchor[] possibleDestinations;

    void Start()
    {
		NetworkManager.Singleton.OnClientConnectedCallback += Singleton_OnClientConnectedCallback;
    }

	IEnumerator WaitAndTeleport()
	{
		yield return new WaitForSeconds(0.5f);
		teleport();
	}

	private void Singleton_OnClientConnectedCallback(ulong obj)
	{
		Debug.Log($"I'am {obj}");
		if(NetworkManager.Singleton.LocalClientId == obj)
		{
			StartCoroutine(WaitAndTeleport());
		}
	}

	public void teleport()
	{
		if(seatAllocator != null)
		{
			possibleDestinations[seatAllocator.getMySeat()].RequestTeleport();
		}
		else
		{
			teleportRandom();
		}
	}

    public void teleportRandom()
    {
        possibleDestinations[UnityEngine.Random.Range(0, possibleDestinations.Length - 1)].RequestTeleport();
    }
}
