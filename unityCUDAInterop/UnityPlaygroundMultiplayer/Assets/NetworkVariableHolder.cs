using Unity.Netcode;
using UnityEngine;
using Unity.Collections;

public class NetworkVariableHolder : NetworkBehaviour
{
    public NetworkVariable<FixedString64Bytes> volumeUUID = new(
        "",
        NetworkVariableReadPermission.Everyone,
        NetworkVariableWritePermission.Owner
    );

    public override void OnNetworkSpawn()
    {
        NetworkManager.Singleton.OnClientConnectedCallback += (clientId) =>
        {
			if (IsOwnedByServer) {
				GetComponent<NetworkObject>().ChangeOwnership(clientId);
				volumeUUID.Value = new FixedString64Bytes("1234567");
			}
			Debug.Log(volumeUUID.Value);
		};
    }
}
