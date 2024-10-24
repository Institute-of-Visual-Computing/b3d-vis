Shader "UnityCudaInterop/TransferFunctionColormapShader"
{
    Properties
    {
        _SingleColor("SingleColor", Color) = (1, 0, 0, 0)
        [NoScaleOffset]
		_MainTex("ColorMaps", 2D) = "white" {}
        _ColorMapsTextureSelectionOffset("ColorMapsTextureSelectionOffset", Float) = 0.2878049
        _ColorMapsTextureHeight("ColorMapsTextureHeight", Float) = 0.01219512
        [ToggleUI]
		_UseColorMap("UseColorMap", Float) = 1
        [NoScaleOffset]
		_TransferFunctionTexture("TransferFunctionTexture", 2D) = "white" {}

    }
    SubShader
    {
        Tags
        {
            "RenderType"="Opaque"
            "BuiltInMaterialType" = "Unlit"
            "Queue"="Geometry"
        }
        LOD 100

        Pass
        {
			Cull Back
			Blend One Zero
			ZTest LEqual
			ZWrite Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
				UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

			Texture2D _MainTex;
			SamplerState my_linear_clamp_sampler;
			Texture2D _TransferFunctionTexture;

            float4 _MainTex_ST;

			float _ColorMapsTextureHeight;
			float _ColorMapsTextureSelectionOffset;
			bool _UseColorMap;

            v2f vert (appdata v)
            {
                v2f o;
				UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
				float2 transferCoord = float2(i.uv.x, 0.5f);
				float4 transferValue = _TransferFunctionTexture.Sample(my_linear_clamp_sampler, i.uv);

				float selectedColorMapCoord =  _ColorMapsTextureSelectionOffset;
				float2 colorMapCoord = float2(transferValue.x, selectedColorMapCoord);
				
				// uv.y is smaller or equal to transferValue.x => 1
				// transferValue.x is greater or equal than uv.y => 1
				fixed4 colorMapValue = _MainTex.Sample(my_linear_clamp_sampler, colorMapCoord) * step(i.uv.y,transferValue.x);

                return colorMapValue;
            }
            ENDCG
        }
    }
}
