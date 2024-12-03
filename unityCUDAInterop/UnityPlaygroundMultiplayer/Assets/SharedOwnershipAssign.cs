using Unity.Netcode;
using UnityEngine;
using XRMultiplayer;

public class SharedOwnershipAssign : NetworkBehaviour
{
	public OwnershipReleaser ownershipReleaser;

	public override void OnGainedOwnership()
	{
		
		if (ownershipReleaser)
		{
			if (OwnerClientId == NetworkManager.LocalClientId)
			{
				Debug.Log($"My CLientId: {NetworkManager.LocalClientId}, ownerClientId: {OwnerClientId}");
				ownershipReleaser.RequestControlsRpc(NetworkManager.LocalClientId);
			}
			
		}
		else
		{
			Debug.LogError("No ownershipReleaser");
		}
	}
}
