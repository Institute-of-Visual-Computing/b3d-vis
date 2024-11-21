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
        NetworkManager.Singleton.OnClientDisconnectCallback += (clientId) =>
        {
            
        };

        if (IsOwnedByServer)
        {
            Debug.Log($"Serveeer: {OwnerClientId}");

        }
        else
        {
            Debug.Log("Client");
        }
    }
}
