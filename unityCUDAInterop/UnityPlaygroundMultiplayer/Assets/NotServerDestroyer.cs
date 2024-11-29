using Unity.Netcode;
using UnityEngine;

public class NotServerDestroyer : NetworkBehaviour
{
	public GameObject[] objectsToDestroy;
	public override void OnNetworkSpawn()
	{
		base.OnNetworkSpawn();
		if (!IsServer)
		{
			foreach (var item in objectsToDestroy)
			{
				Destroy(item);
			}
		}
	}
}
