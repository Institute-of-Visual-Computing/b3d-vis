using System.Runtime.InteropServices;
using UnityEngine;

namespace Utilities
{
    public class ParticlesDrawer : MonoBehaviour
    {
        [SerializeField] private Material _material;
        private ComputeBuffer _particlesComputeBuffer;
        private int _nbrParticles;
        private float _particlesSizeForRender;
        
        private static readonly int _particlesProp = Shader.PropertyToID("particles");
        private static readonly int _sizeParticles = Shader.PropertyToID("sizeParticles");

        private void Awake()
        {
            if (_material == null)
            {
                Debug.LogError("Set a material in inspector for ParticlesDrawer in " + gameObject.name);
                return;
            }
        }
        

        private void OnPostRender()
        {
            _material.SetPass(0);
            _material.SetBuffer(_particlesProp, _particlesComputeBuffer);
            // _particlesComputeBuffer.GetData(x,0,0,1);
            
            _material.SetFloat(_sizeParticles, _particlesSizeForRender);
            Graphics.DrawProceduralNow(MeshTopology.Points, _nbrParticles, 1);
        }

        private void OnDestroy()
        {
            if (_particlesComputeBuffer != null)
            {
                _particlesComputeBuffer.Release();
            }
        }
    }
}
