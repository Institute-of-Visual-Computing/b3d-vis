using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using UnityEngine;

namespace ActionUnity
{
    public class ActionUnityTestOptixOwl : ActionUnity
    {
        const string _dllOptixActions = "OptixAction";

        [DllImport(_dllOptixActions)]
        private static extern IntPtr createActionTestOpixOwl(IntPtr texturePtr, int width, int height, int depth);

        [DllImport(_dllOptixActions)]
        private static extern IntPtr setRenderingData(IntPtr actionObject, IntPtr renderingData);

        
        [DllImport(_dllOptixActions)]
        private static extern IntPtr setTexture(IntPtr actionObject, IntPtr texturePtr, int width, int height, int depth);

        /// <summary>
        /// create a pointer to actionSampleTexture object that has been created in native plugin
        /// </summary>
        /// <param name="texture">texture that will be used in interoperability</param>
        public ActionUnityTestOptixOwl(Texture texture, int volumeDepth = 1) : base()
        {
            _actionPtr = createActionTestOpixOwl(texture.GetNativeTexturePtr(), texture.width, texture.height, volumeDepth);
        }

        public void setRenderingDataForObject(IntPtr renderingData)
        {
            setRenderingData(_actionPtr, renderingData);
        }

        public void setTextureForObject(Texture texture, int width, int height, int depth)
        {
            setTexture(_actionPtr, texture.GetNativeTexturePtr(), width, height, depth);
        }
    }
}
