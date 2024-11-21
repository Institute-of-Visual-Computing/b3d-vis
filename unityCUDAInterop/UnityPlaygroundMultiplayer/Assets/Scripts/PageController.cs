using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PageController : MonoBehaviour
{
    public GameObject page1, page2, page3, page4,page5,page6,page7,page8,page9,page10,page11;
    private List<GameObject> pagelist;

    public int curPage=0;

    public int maxPage;
    // Start is called before the first frame update
    void Start()
    {
        pagelist = new List<GameObject>();
        pagelist.Add(page1);
        pagelist.Add(page2);
        pagelist.Add(page3);
        pagelist.Add(page4);
        pagelist.Add(page5);
        pagelist.Add(page6);
        pagelist.Add(page7);
        pagelist.Add(page8);
        pagelist.Add(page9);
        pagelist.Add(page10);
        pagelist.Add(page11);
        foreach (GameObject page in pagelist)
        {
            page.SetActive(false);
        }
        page1.SetActive(true);
    }

    void setupPages()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void nextPage()
    {
        if (curPage < pagelist.Count)
        {
            curPage++;
            foreach (GameObject page in pagelist)
            {
                page.SetActive(false);
            }
            pagelist[curPage].SetActive(true);
        }
    }

    public void previousPage()
    {
        if (curPage > 0)
        {
            curPage--;
            foreach (GameObject page in pagelist)
            {
                page.SetActive(false);
            }

            pagelist[curPage].SetActive(true);
        }
    }
}
