using System;
using UnityEngine;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering;

class B3DRenderPassData
{
    public IntPtr pluginEventFuncPtr;
    public IntPtr pluginDataPtr;
    public int eventId;
}

public class B3DRenderPass : ScriptableRenderPass
{
    IntPtr pluginEventFuncPtr;
    IntPtr pluginDataPtr;
    int eventId;

    public B3DRenderPass(IntPtr pluginEventFuncPtr, int eventId, IntPtr pluginDataPtr)
    {
        this.pluginEventFuncPtr = pluginEventFuncPtr;
        this.eventId = eventId;
        this.pluginDataPtr = pluginDataPtr;
    }

    public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
    {
        // var graphBuilder = renderGraph.AddComputePass<B3DRenderPassData>("B3dPass", out passData);
        var builder = renderGraph.AddRasterRenderPass<B3DRenderPassData>("B3dPass", out var passData);
        passData.pluginEventFuncPtr = pluginEventFuncPtr;
        passData.eventId = eventId;
        passData.pluginDataPtr = pluginDataPtr;

        builder.AllowPassCulling(false);

        builder.SetRenderFunc<B3DRenderPassData>((B3DRenderPassData passData, RasterGraphContext context) =>
        {
            // Do something
            context.cmd.IssuePluginEventAndData(passData.pluginEventFuncPtr, passData.eventId, passData.pluginDataPtr);
        });
        builder.Dispose();
        /*
        graphBuilder.SetRenderFunc<B3DRenderPassData>((B3DRenderPassData passData, ComputeGraphContext context) =>
        {
            // Do something
            context.cmd.IssuePluginEventAndData(passData.pluginEventFuncPtr, passData.eventId, passData.pluginDataPtr);
            
        });
        graphBuilder.Dispose();
        */
    }
}