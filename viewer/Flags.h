#pragma once

#include <cstdint>

template <typename FlagBitsType>
struct Flags
{
	using FlagsType = uint16_t;
	FlagsType flags;

	Flags(const FlagBitsType flagBits)
	{
		flags = static_cast<FlagsType>(flagBits);
	}

	Flags& operator|=(FlagBitsType r)
	{
		flags |= static_cast<uint16_t>(r);
		return *this;
	}

	Flags& operator|=(Flags r)
	{
		flags |= static_cast<uint16_t>(r.flags);
		return *this;
	}

	Flags& operator|(Flags r)
	{
		flags |= static_cast<uint16_t>(r.flags);
		return *this;
	}


	FlagBitsType operator&(FlagBitsType r)
	{
		return static_cast<FlagBitsType>(flags & static_cast<FlagsType>(r));
	}


	bool operator==(const Flags& other) const
	{
		return flags == other.flags;
	}
};
