#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include "SharedStructs.h"

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

	};

	using Schema = std::unordered_map<std::string, size_t>;

	struct SchemaData
	{
		Schema schema{};
		size_t schemaContentSize{};
	};


#define SCHEMA_ENTRY(keyValue, memberName, parentStruct)                                                               \
	{                                                                                                                  \
		keyValue, offsetof(parentStruct, memberName)                                                                   \
	}

	static const SchemaData schemaData_0{ {
											  SCHEMA_ENTRY("rendererInitializationInfo", rendererInitializationInfo, RenderingData),
											  SCHEMA_ENTRY("view", view, RenderingData),
											  SCHEMA_ENTRY("renderTargets", renderTargets, RenderingData),
											  SCHEMA_ENTRY("synchronization", synchronization, RenderingData),
											  SCHEMA_ENTRY("volumeTransform", volumeTransform, RenderingData),
											  SCHEMA_ENTRY("transferOffset", transferOffset, RenderingData),
										  },
										  sizeof(RenderingData) };


	class RenderingDataBuffer
	{
	private:
		SchemaData schemaData_{};

		std::byte* dataPtr_{ nullptr };
		std::vector<std::byte> data_{};

		bool isBufferOwned_{ false };

		uint64_t bufferCount_{ 1 };
		inline auto getSchemaEntry(const std::string& key) const
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

		template <typename T>
		auto get(const std::string& key) -> T*
		{
			const auto schemaEntry = getSchemaEntry(key);
			if (schemaData_.schema.end() == schemaEntry)
			{
				return nullptr;
			}
			return reinterpret_cast<T*>(&dataPtr_[schemaEntry->second]);
		}
	/*
		template <typename T>
		auto getOpt(const std::string& key) -> std::optional<T&>
		{
			const auto schemaEntry = getSchemaEntry(key);
			if (schemaData_.schema.end() == schemaEntry)
			{
				return std::nullopt;
			}
			return static_cast<T>(dataPtr_[schemaEntry->second]);
		}

		template <typename T>
		auto tryGet(const std::string& key, T& t) -> bool
		{
			const auto schemaEntry = getSchemaEntry(key);
			if (schemaData_.schema.end() == schemaEntry)
			{
				return false;
			}
			t = static_cast<T>(dataPtr_[schemaEntry->second]);
			return true;
		}

		

		template <typename T>
		auto operator[](const std::string& key) -> T&
		{
			return static_cast<T>(dataPtr_[schemaData_.schema[key]]);
		}
		*/
	};

	struct RenderingDataWrapper
	{
		RenderingData data{};
		RenderingDataBuffer buffer;
		RenderingDataWrapper() : buffer{ schemaData_0, 1, static_cast<void*>(&data) }
		{
			
		}
	};
} // namespace b3d::renderer
