using Unity.Netcode;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEngine;
using XRMultiplayer;

public class OwnershipReleaser : NetworkBehaviour
{
	bool allowOtherOwner = false;

	public NetworkObject[] objectsToChangeOwnership;

	public void forceServerOwnership()
	{
		if(!IsServer)
		{
			return;
		}
		allowOtherOwner = false;
		GetComponent<NetworkObject>().RemoveOwnership();
		foreach (var item in objectsToChangeOwnership)
		{
			item.RemoveOwnership();
		}
		// TODO: Force get ownership for this object
	}

	public void releaseServerOwnerShip()
	{
		allowOtherOwner = true;
	}

	[Rpc(SendTo.Server)]
	public void RequestControlsRpc(ulong clientId)
	{
		// Server is always allowed to take ownership.
		if(allowOtherOwner || clientId == NetworkManager.ServerClientId)
		{
			var nwobj = GetComponent<NetworkObject>();
			if(OwnerClientId != clientId)
			{
				nwobj.ChangeOwnership(clientId);
			}
			foreach (var item in objectsToChangeOwnership)
			{
				item.ChangeOwnership(clientId);
			}
		}
	}
}
