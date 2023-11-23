using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{
    public class OptixSourceVolumeActions : ActionUnity
    {
        const string _dllOptixActions = "OptixSourceVolumeActions";

        [DllImport(_dllOptixActions, CharSet = CharSet.Ansi)]
        private static extern IntPtr createActionSourceVolumeTest(IntPtr texturePtr, int width, int height, int depth, string fitsFilePath, string catalogFilePath);

        [DllImport(_dllOptixActions)]
        private static extern IntPtr setRenderingDataSourceVolumeTest(IntPtr actionObject, IntPtr renderingData);

        
        [DllImport(_dllOptixActions)]
        private static extern IntPtr setTextureSourceVolumeTest(IntPtr actionObject, IntPtr texturePtr, int width, int height, int depth);

        /// <summary>
        /// create a pointer to actionSampleTexture object that has been created in native plugin
        /// </summary>
        /// <param name="texture">texture that will be used in interoperability</param>
        public OptixSourceVolumeActions(Texture texture, int volumeDepth, string fitsFilePath, string catalogFilePath) : base()
        {
            _actionPtr = createActionSourceVolumeTest(texture.GetNativeTexturePtr(), texture.width, texture.height, volumeDepth, fitsFilePath, catalogFilePath);
        }

        public void setRenderingDataForObject(IntPtr renderingData)
        {
            setRenderingDataSourceVolumeTest(_actionPtr, renderingData);
        }

        public void setTextureForObject(Texture texture, int width, int height, int depth)
        {
            setTextureSourceVolumeTest(_actionPtr, texture.GetNativeTexturePtr(), width, height, depth);
        }
    }
}
