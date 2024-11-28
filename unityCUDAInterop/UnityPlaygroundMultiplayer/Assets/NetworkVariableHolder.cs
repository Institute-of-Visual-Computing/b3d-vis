using Unity.Netcode;
using Unity.Collections;
using System.Collections.Generic;

public class NetworkVariableHolder : NetworkBehaviour
{
	


	public ColoringChanger coloringChanger;

	private void Awake()
	{
		
	}

	public override void OnNetworkSpawn()
    {
    }
}
