using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

public class SourceLoader : MonoBehaviour
{
    const string _dllFile = "OptixActions";
    [DllImport(_dllFile, CharSet = CharSet.Ansi, EntryPoint = "loadSources")]
    private static extern int loadSources(string fitsFile, string catalogFile);

    public string catalogFilePath = "D:/data/work/b3d/n4565/sofia_output/outname_cat.xml";

    public string fitsFilepath = "D:/data/work/b3d/n4565/n4565_lincube_big.fits";

    async void Start()
    {
        int one = await loadSourcesAsync();
        Debug.Log("All Done!");
        Debug.Log(one);
    }

    async Task<int> loadSourcesAsync()
    {
        var resultTask = Task<int>.Factory.StartNew(() => {
            return loadSources(fitsFilepath, catalogFilePath);
        });
        await resultTask;
        return resultTask.Result;
    }

    private void Update()
    {

    }
}
