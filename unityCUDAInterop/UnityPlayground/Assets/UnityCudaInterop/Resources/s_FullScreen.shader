
Shader "UnityCudaInterop/Fullscreen_Quad"
{
	Properties
	{
		_MainTex ("Texture", 2DArray) = "white" {}
	}

	SubShader
	{
		Cull Off
		ZWrite Off
        ColorMask RGBA
		// ZClip False // Sets the depth clip mode to clamp. Fragments closer than the near plane are at the near plane exactly, and fragments further away than the far plane are at the far plane exactly.
		// ZTest Always // No depth testing occurs. Draw all geometry, regardless of distance.

		Tags { "RenderType"="Transparent" }

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
				float2 uv     : TEXCOORD0;
				uint   id     : SV_VertexID;
				UNITY_VERTEX_INPUT_INSTANCE_ID
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
			};

            UNITY_DECLARE_TEX2DARRAY(_MainTex);
            float4 _MainTex_ST;
            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);


			v2f vert(appdata v)
			{
				v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

				// 0, 2, 1
				// 2, 3, 1
				float4 verts[4] = {
					float4(-1,  1, 0, 1),  // l, t
					float4( 1,  1, 0, 1),  // r, t
					float4(-1, -1, 0, 1),  // l, b
					float4( 1, -1, 0, 1),  // r, b
				};
				o.vertex = verts[v.id];
				o.uv = v.uv;
				return o;
			}

			fixed4 frag(v2f i /*, out float depth : SV_Depth*/) : SV_Target
			{
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
				float3 texPos = float3(i.uv, unity_StereoEyeIndex);

				#if !UNITY_UV_STARTS_AT_TOP
                    texPos.y = 1.0f - texPos.y;
                #endif
				
                fixed4 col = UNITY_SAMPLE_TEX2DARRAY(_MainTex, texPos);
				if(col.w < 0.0001f)
                {
                    float abc = 1;
                    discard;
                }
				return col;
			}
			ENDCG
		}
	}
}
