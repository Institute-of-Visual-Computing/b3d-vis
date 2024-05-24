Shader "UnityCudaInterop/ProjectToObject"
{
    Properties
    {
        _MainTex ("Texture", 2DArray) = "white" {}
    }
    SubShader
	{
        ZWrite Off
        ColorMask RGBA
		Cull Off

		Tags { "Queue" = "Transparent" "RenderType"="Transparent" "ForceNoShadowCasting"="True" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma require 2darray
            #pragma enable_d3d11_debug_symbols

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
				UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float3 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float4 screenPos : TEXCOORD1;
                UNITY_VERTEX_OUTPUT_STEREO
            };
			
            UNITY_DECLARE_TEX2DARRAY(_MainTex);
            float4 _MainTex_ST;
            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv.xy = TRANSFORM_TEX(v.uv, _MainTex);
                o.screenPos = ComputeScreenPos(o.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

				float3 texPos = float3((i.screenPos.xy / i.screenPos.w), unity_StereoEyeIndex);

				#if !UNITY_UV_STARTS_AT_TOP
                    texPos.y = 1.0f - texPos.y;
                #endif
				
                fixed4 col = UNITY_SAMPLE_TEX2DARRAY(_MainTex, texPos);
				float lineDepth = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, texPos));
				if((col.a)<0.001f)
                {
					col.r = 1;
                    float abc = lineDepth;
					discard;
                }

                return col;
            }
            ENDCG
        }
    }
}
