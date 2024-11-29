using Unity.Netcode;
using UnityEngine;

public class NotServerHider : NetworkBehaviour
{
	public override void OnNetworkSpawn()
	{
		base.OnNetworkSpawn();
		if(!IsServer)
		{
			Destroy(gameObject);
		}
	}
}
