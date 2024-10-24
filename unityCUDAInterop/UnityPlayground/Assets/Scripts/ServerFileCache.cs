using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.XR.ARSubsystems;


public class ServerFileCache : MonoBehaviour
{

	public ServerClient serverClient;

	// uid, path
	Dictionary<string, string> cache = new();

	string cachePath; 


	private void Awake()
	{
		cachePath = Path.Combine(Application.persistentDataPath, "b3dFileCache");
		serverClient.FileDownloadedEvent += fileDownloadFinished;
		if(!Directory.Exists(cachePath))
		{
			var dirInfo = Directory.CreateDirectory(cachePath);
			if(!dirInfo.Exists)
			{
				cachePath = Application.persistentDataPath;
			}
		}
	}

	protected void fileDownloadFinished(string uuid, string path)
	{
		cache.Add(uuid, path);
	}

	public Task<Tuple<string, string>> downloadFile(string uuid)
	{
		if(cache.ContainsKey(uuid))
		{
			return Task.FromResult(Tuple.Create(uuid, cache[uuid]));
		}
		else
		{
			return serverClient.downloadFile(cachePath, uuid);
		}
	}
}
