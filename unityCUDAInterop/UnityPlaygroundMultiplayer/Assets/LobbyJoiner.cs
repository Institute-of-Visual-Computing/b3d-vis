using Unity.Netcode;
using UnityEngine;
using XRMultiplayer;
using System.Collections;
using Unity.Services.Lobbies.Models;

public class LobbyJoiner : MonoBehaviour
{
	// Start is called once before the first execution of Update after the MonoBehaviour is created
	void Start()
	{
		StartCoroutine(WaitAndConnect());	
	}

	IEnumerator WaitAndConnect()
	{
		yield return new WaitForSeconds(1);
		bool lobbyFound = false;
		var lobbiesTask = LobbyManager.GetLobbiesAsync();

		yield return new WaitUntil(() => lobbiesTask.IsCompleted);
		var lobbies = lobbiesTask.Result;
		if (lobbies.Results != null || lobbies.Results.Count > 0)
		{
			foreach (var lobby in lobbies.Results)
			{
				if (lobby.Name == "B3D")
				{
					if (LobbyManager.CanJoinLobby(lobby))
					{
						lobbyFound = true;
						XRINetworkGameManager.Instance.JoinLobbySpecific(lobby);
						yield return null;
					}
				}
			}
		}

		if (!lobbyFound)
		{
			XRINetworkGameManager.Instance.CreateNewLobby("B3D");
		}
	}


    // Update is called once per frame
    void Update()
    {
        
    }

	void OnConnected(bool connected)
	{
		if (connected)
		{
			XRINetworkGameManager.Connected.Unsubscribe(OnConnected);
		}
	}
}
