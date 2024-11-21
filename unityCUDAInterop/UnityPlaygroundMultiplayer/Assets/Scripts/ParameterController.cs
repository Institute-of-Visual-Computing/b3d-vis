using System.Collections;
using System.Collections.Generic;
using MixedReality.Toolkit.UX;
using Unity.XR.CoreUtils;
using UnityEngine;

public class ParameterController : MonoBehaviour
{
    public PressableButton invertButton;
    private bool invertValue, default_invertValue;

    public MixedReality.Toolkit.UX.Slider orderSlider,thresholdSlider;
    private float order,default_order;
    private float threshold,default_threshold;

    public GameObject statisticsMenu;
    public GameObject flagMenu,ripple_filter_menu,scaleNoise_mode_menu,scaleNoise_statistic_menu,scFind_fluxRange_menu,scFind_kernelsXY_menu,
    scFind_kernelsZ_menu,scFind_statistic_menu,threshold_fluxRange_menu,threshold_mode_menu,threshold_statistic_menu,reliability_parameters_menu;
    private string statistic, default_statistic="mad";
    // Start is called before the first frame update
    void Start()
    {
        default_invertValue = invertButton.IsToggled;
        default_order = orderSlider.Value;
        default_threshold = thresholdSlider.Value;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void enable_statistics_menu()
    {
        statisticsMenu.SetActive(true);
    }
    public void disable_statistics_menu()
    {
        statisticsMenu.SetActive(false);
    }
    
    public void enable_flag_menu()
    {
        flagMenu.SetActive(true);
    }
    public void disable_flag_menu()
    {
        flagMenu.SetActive(false);
    }
    
    public void enable_ripple_filter_menu()
    {
        ripple_filter_menu.SetActive(true);
    }
    public void disable_ripple_filter_menu()
    {
        ripple_filter_menu.SetActive(false);
    }
    
    public void enable_scaleNoise_mode_menu()
    {
        scaleNoise_mode_menu.SetActive(true);
    }
    public void disable_scaleNoise_mode_menu()
    {
        scaleNoise_mode_menu.SetActive(false);
    }
    public void enable_scaleNoise_statistic_menu()
    {
        scaleNoise_statistic_menu.SetActive(true);
    }
    public void disable_scaleNoise_statistic_menu()
    {
        scaleNoise_statistic_menu.SetActive(false);
    }
    public void enable_scFind_fluxRange_menu()
    {
        scFind_fluxRange_menu.SetActive(true);
    }
    public void disable_scFind_fluxRange_menu()
    {
        scFind_fluxRange_menu.SetActive(false);
    }
    public void enable_scFind_kernelsXY_menu()
    {
        scFind_kernelsXY_menu.SetActive(true);
    }
    public void disable_scFind_kernelsXY_menu()
    {
        scFind_kernelsXY_menu.SetActive(false);
    }
    public void enable_scFind_kernelsZ_menu()
    {
        scFind_kernelsZ_menu.SetActive(true);
    }
    public void disable_scFind_kernelsZ_menu()
    {
        scFind_kernelsZ_menu.SetActive(false);
    }
    public void enable_scFind_statistic_menu()
    {
        scFind_statistic_menu.SetActive(true);
    }
    public void disable_scFind_statistic_menu()
    {
        scFind_statistic_menu.SetActive(false);
    }
    public void enable_threshold_fluxRange_menu()
    {
        threshold_fluxRange_menu.SetActive(true);
    }
    public void disable_threshold_fluxRange_menu()
    {
        threshold_fluxRange_menu.SetActive(false);
    }
    public void enable_threshold_mode_menu()
    {
        threshold_mode_menu.SetActive(true);
    }
    public void disable_threshold_mode_menu()
    {
        threshold_mode_menu.SetActive(false);
    }
    public void enable_threshold_statistic_menu()
    {
        threshold_statistic_menu.SetActive(true);
    }
    public void disable_threshold_statistic_menu()
    {
        threshold_statistic_menu.SetActive(false);
    }
    public void enable_reliability_parameters_menu()
    {
        reliability_parameters_menu.SetActive(true);
    }
    public void disable_reliability_parameters_menu()
    {
        reliability_parameters_menu.SetActive(false);
    }
    

    public void resetValues()
    {
        invertValue = default_invertValue;
        invertButton.ForceSetToggled(default_invertValue);
        order = default_order;
        orderSlider.Value = default_order;
        threshold = default_threshold;
        thresholdSlider.Value = default_threshold;
        statistic = default_statistic;
    }
}
