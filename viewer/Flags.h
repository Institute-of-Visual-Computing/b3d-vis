#pragma once

#include <cstdint>

template <typename FlagBitsType>
	requires(std::is_enum_v<FlagBitsType>)
struct Flags
{
	using FlagsType = std::underlying_type_t<FlagBitsType>;

	Flags()
	{
	}

	Flags(const FlagBitsType& flagBits)
	{
		flags = static_cast<FlagsType>(flagBits);
	}

	auto operator|=(const FlagBitsType& r) -> Flags&
	{
		flags |= static_cast<FlagsType>(r);
		return *this;
	}

	auto operator|=(const Flags& r) -> Flags&
	{
		flags |= static_cast<FlagsType>(r.flags);
		return *this;
	}

	auto operator|(const Flags& r) const -> Flags
	{
		Flags f;
		f.flags = flags | static_cast<FlagsType>(r.flags);
		return f;
	}

	

	auto operator&(const Flags& r) const -> Flags
	{
		return static_cast<FlagBitsType>(flags & static_cast<FlagsType>(r.flags));
	}

	auto operator~() -> Flags&
	{

		flags = ~flags;
		return *this;
	}

	auto flip(const FlagBitsType& flagBit) -> Flags&
	{
		auto bit = static_cast<FlagsType>(flagBit);
		flags = (flags & (~bit)) | (~(flags & bit)) & bit;
		return *this;
	}

	[[nodiscard]] auto containsBit(const FlagBitsType& flagBit) const -> bool
	{
		return static_cast<bool>(flags & static_cast<FlagsType>(flagBit));
	}

	auto setBit(const FlagBitsType& flagBit) -> void
	{
		flags |= flagBit;
	}

	auto unsetBit(const FlagBitsType& flagBit) -> void
	{
		flags &= ~flagBit;
	}

	bool operator==(const Flags& other) const
	{
		return flags == other.flags;
	}

private:
	FlagsType flags{};
};

//template <typename FlagBitsType>
//	requires(std::is_enum_v<FlagBitsType> && requires(FlagBitsType f) { Flags<FlagBitsType>{ f }; })
//inline auto operator|(const FlagBitsType& a, const FlagBitsType& b) -> Flags<FlagBitsType>
//{
//	auto flags = Flags<FlagBitsType>{ a };
//	flags |= b;
//	return flags;
//}
