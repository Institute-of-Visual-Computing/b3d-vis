using System;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit.Locomotion.Teleportation;
using XRMultiplayer;

public class RandomTeleporter : MonoBehaviour
{
    public TeleportationAnchor[] possibleDestinations;

    void OnEnable()
    {
        XRINetworkGameManager.Connected.Subscribe(TeleportOnConnect);
    }

    public void teleportRandom()
    {
        possibleDestinations[UnityEngine.Random.Range(0, possibleDestinations.Length)].RequestTeleport();
    }

    protected virtual void TeleportOnConnect(bool online)
    {
        if(online)
        {
            teleportRandom();
        }
    }
}
