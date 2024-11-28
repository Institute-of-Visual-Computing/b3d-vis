using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;




public class SoFiASearchRequest
{
    [JsonProperty("input.data")]
    public string inputData { get; set; }

    [JsonProperty("input.region")]
    public string inputRegion { get; set; }
}

public class SoFiASearchResult
{
    public bool wasSuccess = false;
    public string message = string.Empty;
    public string outputDirectory;
}

public class SoFiaSearch
{
    public SoFiASearchRequest request;
    public SoFiASearchResult result;
    public string search_hash;
}


public class SofiaServerConnector : MonoBehaviour
{
    bool shouldStopSearch = false;

    public Transform subRegionTransform;

    public Vector3 lowerBoxPos;
    public Vector3 upperBoxPos;

    public void RunSofiaSearch()
    {
        StartCoroutine(SofiaSearch());
    }

    IEnumerator SofiaSearch()
    {

        shouldStopSearch = false;
        SoFiaSearch search = new SoFiaSearch();
        search.result = new();
        search.request = new SoFiASearchRequest();
        search.request.inputData = "n4565/n4565_lincube_big.fits"; // 1024, 1024, 448
        search.request.inputRegion = "0,200,0,200,0,200";
        Vector3 maxBounds = new(1024, 1024, 448);

        Vector3.Scale(maxBounds, subRegionTransform.localScale);

        Vector3 lower = new(-.5f, -.5f, -.5f ) ;
        Vector3 upper = new( .5f, .5f, .5f );
        Vector3 lowerPos = Vector3.Max(subRegionTransform.transform.localPosition - subRegionTransform.localScale * 0.5f, lower) + upper;
        Vector3 upperPos = Vector3.Min(subRegionTransform.transform.localPosition + subRegionTransform.localScale * 0.5f, upper) + upper;

        lowerBoxPos = Vector3.Scale(lowerPos, maxBounds);
        upperBoxPos = Vector3.Scale(upperPos, maxBounds);

        search.request.inputRegion = $"{lowerBoxPos.x},{upperBoxPos.x},{lowerBoxPos.y},{upperBoxPos.y},{lowerBoxPos.z},{upperBoxPos.z}";

        using (UnityWebRequest www = UnityWebRequest.Post("http://127.0.0.1:8080/start", JsonConvert.SerializeObject(search.request), "application/json"))
        {
            yield return www.SendWebRequest();

            if(www.result != UnityWebRequest.Result.ProtocolError && www.result != UnityWebRequest.Result.Success)
            {
                search.result.wasSuccess = false;
                search.result.message = www.error;
                Debug.Log(www.error);
                yield break; 
            }
            
            if (www.responseCode == 200)
            {
                var jsResultObject = JObject.Parse(www.downloadHandler.text);
                search.search_hash = jsResultObject["search_hash"].Value<string>();
            }
            else
            {
                search.result.wasSuccess = false;
                search.result.message = www.downloadHandler.text;
                yield break;
            }
        }

        bool stop = false;
        JObject obj = new JObject();
        obj["search_hash"] = search.search_hash;
        while(!stop && !shouldStopSearch)
        {
            using (UnityWebRequest www = UnityWebRequest.Post("http://127.0.0.1:8080/result", JsonConvert.SerializeObject(obj), "application/json"))
            {
                yield return www.SendWebRequest();
                if (www.result != UnityWebRequest.Result.ProtocolError && www.result != UnityWebRequest.Result.Success)
                {
                    search.result.wasSuccess = false;
                    search.result.message = www.error;
                    Debug.Log(www.error);
                    stop = true;
                    break;
                }
                if (www.responseCode == 503)
                {
                    Debug.Log("running");
                    yield return new WaitForSeconds(0.2f);
                    continue;
                }
                else if(www.responseCode == 200)
                {
                    search.result.wasSuccess = true;
                    search.result.message = www.downloadHandler.text;
                    
                    Debug.Log(search.result.message);
                    stop = true;
                    break;
                }
                else if(www.responseCode == 400)
                {
                    search.result.wasSuccess = false;
                    search.result.message = www.downloadHandler.text;
                    stop = true;
                    break;
                }
                else
                {
                    search.result.wasSuccess = false;
                    search.result.message = "Unknown error";
                    stop = true;
                    break;
                }
            }
        }
    }
}
