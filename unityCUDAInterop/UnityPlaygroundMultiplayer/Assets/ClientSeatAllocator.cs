using JetBrains.Annotations;
using MixedReality.Toolkit.SpatialManipulation;
using NUnit.Framework;
using System;
using Unity.Netcode;
using UnityEditor.PackageManager;
using UnityEngine;

public struct ClientSeat : INetworkSerializable, System.IEquatable<ClientSeat>
{
	public ulong playerId;
	public int seat;

	public void NetworkSerialize<T>(BufferSerializer<T> serializer) where T : IReaderWriter
	{
		if (serializer.IsReader)
		{
			var reader = serializer.GetFastBufferReader();
			reader.ReadValueSafe(out playerId);
			reader.ReadValueSafe(out seat);
		}
		else
		{
			var writer = serializer.GetFastBufferWriter();
			writer.WriteValueSafe(playerId);
			writer.WriteValueSafe(seat);
		}
	}

	public bool Equals(ClientSeat other)
	{
		return playerId == other.playerId && seat == other.seat;
	}
}

public class ClientSeatAllocator : NetworkBehaviour
{
	public NetworkList<ClientSeat> m_SeatAssignments = new();

	bool[] occupiedSeats = new bool[3]{ false, false, false };

	int getSeat(ulong clientId)
	{
		Debug.Log($"I'am {clientId}. Do I have a seat? {findClientIdIdxInSeats(clientId)}");
		if(findClientIdIdxInSeats(clientId) > -1)
		{
			return m_SeatAssignments[findClientIdIdxInSeats(clientId)].seat;
		}
		return 0;
	}

	public int getMySeat()
	{
		return getSeat(NetworkManager.Singleton.LocalClientId);
	}

	public void Start()
	{
		NetworkManager.Singleton.OnClientConnectedCallback += Singleton_OnClientConnectedCallback;
		NetworkManager.Singleton.OnClientDisconnectCallback += Singleton_OnClientDisconnectCallback;
	}

	public override void OnNetworkSpawn()
	{
		Debug.Log("onNetworkSpawn");
	}
	int findClientIdIdxInSeats(ulong clientId)
	{
		int clientIdIdx = -1;
		for (int i = 0; i < m_SeatAssignments.Count; i++)
		{
			if (m_SeatAssignments[i].playerId == clientId)
			{
				clientIdIdx = i;
				break;
			}
		}
		return clientIdIdx;
	}

	private void Singleton_OnClientDisconnectCallback(ulong clientId)
	{
		if (!IsServer)
		{
			return;
		}
		var clientIdIdx = findClientIdIdxInSeats(clientId);
		if (clientIdIdx > -1)
		{
			if(m_SeatAssignments[clientIdIdx].seat < occupiedSeats.Length)
			{
				occupiedSeats[m_SeatAssignments[clientIdIdx].seat] = false;
			}
			m_SeatAssignments.RemoveAt(clientIdIdx);
		}
	}

	private void Singleton_OnClientConnectedCallback(ulong clientId)
	{
		if(!IsServer)
		{
			return;
		}
		var clientIdIdx = findClientIdIdxInSeats(clientId);
		if (clientIdIdx < 0)
		{
			ClientSeat seat = new();
			seat.playerId = clientId;
			seat.seat = Array.FindIndex(occupiedSeats, x => x == false);
			if(seat.seat < 0)
			{
				seat.seat = occupiedSeats.Length;
			}
			else
			{
				occupiedSeats[seat.seat] = true;
			}
			m_SeatAssignments.Add(seat);
		}
	}
}
