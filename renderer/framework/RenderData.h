#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "Common.h"
#include "SharedRenderingStructs.h"

namespace b3d::renderer
{
	struct RenderingData
	{
		RendererInitializationInfo rendererInitializationInfo;
		View view;
		RenderTargets renderTargets;
		Synchronization synchronization;
		VolumeTransform volumeTransform;
		float transferOffset{ 0.5f };
		ExternalTexture colorMapTexture;
		ColoringInfo coloringInfo;
		ColorMapInfos colorMapInfos;
		ExternalTexture transferFunctionTexture;
		FoveatedRenderingControl foveatedRenderingControl;
		tools::renderer::nvdb::RuntimeVolumeData runtimeVolumeData;
	};

	using Schema = std::unordered_map<std::string_view, size_t>;

	struct SchemaData
	{
		Schema schema{};
		size_t schemaContentSize{};
	};
#pragma warning(push, 0)
#define SCHEMA_ENTRY(keyValue, memberName, parentStruct) { keyValue, offsetof(parentStruct, memberName) }

	static const SchemaData schemaData_0{
		{ SCHEMA_ENTRY("rendererInitializationInfo", rendererInitializationInfo, RenderingData),
		  SCHEMA_ENTRY("view", view, RenderingData), SCHEMA_ENTRY("renderTargets", renderTargets, RenderingData),
		  SCHEMA_ENTRY("synchronization", synchronization, RenderingData),
		  SCHEMA_ENTRY("volumeTransform", volumeTransform, RenderingData),
		  SCHEMA_ENTRY("transferOffset", transferOffset, RenderingData),
		  SCHEMA_ENTRY("colorMapTexture", colorMapTexture, RenderingData),
		  SCHEMA_ENTRY("coloringInfo", coloringInfo, RenderingData),
		  SCHEMA_ENTRY("colorMapInfos", colorMapInfos, RenderingData),
		  SCHEMA_ENTRY("transferFunctionTexture", transferFunctionTexture, RenderingData),
		  SCHEMA_ENTRY("foveatedRenderingControl", foveatedRenderingControl, RenderingData),
		  SCHEMA_ENTRY("runtimeVolumeData", runtimeVolumeData, RenderingData) },
		sizeof(RenderingData)
	};
#undef SCHEMA_ENTRY
#pragma warning(pop)
	class RenderingDataBuffer
	{
	private:
		SchemaData schemaData_{};

		std::byte* dataPtr_{ nullptr };
		std::vector<std::byte> data_{};

		bool isBufferOwned_{ false };

		uint64_t bufferCount_{ 1 };
		inline auto getSchemaEntry(const std::string_view key) const
		{
			return schemaData_.schema.find(key);
		}


	public:
		RenderingDataBuffer()
		{
		}

		RenderingDataBuffer(const SchemaData& schemaData, const int bufferCount = 1, const bool createBuffer = true)
			: schemaData_{ schemaData }, bufferCount_{ static_cast<uint64_t>(bufferCount) }
		{
			if (createBuffer)
			{
				isBufferOwned_ = true;
				data_.resize(bufferCount_ * schemaData_.schemaContentSize);
				dataPtr_ = data_.data();
			}
		}

		~RenderingDataBuffer()
		{
			dataPtr_ = nullptr;
		}

		RenderingDataBuffer(const SchemaData& schemaData, const int bufferCount, void* buffer)
			: RenderingDataBuffer(schemaData, bufferCount, false)
		{
			dataPtr_ = static_cast<std::byte*>(buffer);
		}

		auto getDataBufferPtr() const -> void*
		{
			return dataPtr_;
		}

		template <typename T>
		inline auto hasKey(const std::string_view key) const -> bool
		{
			return get<T>(key) != nullptr;
		}

		template <typename T>
		auto get(const std::string_view key) -> T*
		{
			const auto schemaEntry = getSchemaEntry(key);
			if (schemaData_.schema.end() == schemaEntry)
			{
				return nullptr;
			}
			return reinterpret_cast<T*>(&dataPtr_[schemaEntry->second]);
		}

		template <typename T>
		auto get(const std::string& key) const -> T*
		{
			const auto schemaEntry = getSchemaEntry(key);
			if (schemaData_.schema.end() == schemaEntry)
			{
				return nullptr;
			}
			return reinterpret_cast<T*>(&dataPtr_[schemaEntry->second]);
		}
	};

	class RenderingDataWrapper
	{
	public:
		RenderingData data{};
		RenderingDataBuffer buffer;
		RenderingDataWrapper() : buffer{ schemaData_0, 1, static_cast<void*>(&data) }
		{
		}
	};
} // namespace b3d::renderer
